"""并发竞态测试：上传与确认发布按数据库提交顺序裁决。

覆盖两类关键交错：

* 新上传先提交 -> 旧确认必须被拒绝，且不产生任何可下载输出；
* 旧确认先提交 -> 已发布文件保持可下载，新批次仅进入待确认。
"""
import threading

import pytest
from fastapi.testclient import TestClient

from app import anonymizer as az
from app import database as db


CSV_HEADER = "受试者号,年龄,邮编,病种"


def _csv_bytes(rows):
    lines = [CSV_HEADER] + [",".join(map(str, r)) for r in rows]
    return "\n".join(lines).encode("utf-8-sig")


def _group(prefix, age, zipc):
    return [(f"{prefix}{i}", age, zipc,
             "糖尿病" if i % 2 == 0 else "高血压") for i in range(3)]


VALID = _group("A", 40, "10000") + _group("B", 50, "10000")
OTHER = _group("C", 60, "30000") + _group("D", 70, "40000")


def _plan_json(prefix="A"):
    rows = az.parse_csv(_csv_bytes(_group(prefix, 40, "10000")))
    return az.plan_to_json(az.find_best_plan(rows))


def _assert_no_residue(db_file):
    with db.session(db_file) as conn:
        rows = list(conn.execute("SELECT * FROM batches"))
    published = [r for r in rows if r["status"] == db.STATUS_PUBLISHED]
    for r in published:
        assert r["output_csv"] is not None, "published 批次缺少输出文件"
        assert r["output_filename"], "published 批次缺少文件名"
    pendings = [r for r in rows if r["status"] == db.STATUS_PENDING]
    assert len(pendings) <= 1, "同一时刻至多一个可确认预览"
    return {r["id"]: r for r in rows}


@pytest.fixture()
def db_file(tmp_path):
    path = tmp_path / "race.db"
    db.init_db(path)
    return path


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import importlib
    import sys
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("ANON_DB_PATH", str(db_file))
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c, db_file


def test_upload_committed_first_blocks_stale_publish(db_file):
    """新上传先提交：发布必须读到 rejected 并拒绝，绝不写入输出。"""
    with db.session(db_file) as conn:
        stale_id = db.insert_pending(conn, "old.csv", 6, _plan_json())

    uploader = db.connect(db_file)
    started = threading.Event()
    finished = threading.Event()
    outcome = {}

    def attempt_publish():
        publisher = db.connect(db_file)
        try:
            started.set()
            db.publish_batch(publisher, stale_id, b"should-never-appear",
                             f"anonymous_batch_{stale_id}.csv")
            outcome["published"] = True
        except db.PreviewInvalidError as exc:
            outcome["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - 记录意外异常供断言
            outcome["unexpected"] = repr(exc)
        finally:
            publisher.close()
            finished.set()

    try:
        upload_txn = db.write_transaction(uploader)
        upload_txn.__enter__()
        uploader.execute(
            "UPDATE batches SET status = 'rejected' WHERE status = 'pending'")
        new_id = uploader.execute(
            "INSERT INTO batches (status, source_filename, row_count, plan_json) "
            "VALUES ('pending', 'new.csv', 6, ?)", (_plan_json("C"),),
        ).lastrowid

        t = threading.Thread(target=attempt_publish)
        t.start()
        assert started.wait(5)
        # 发布应阻塞在写锁上，而不是穿越“读状态 → 写状态”的窗口
        assert not finished.wait(0.3), "发布未等待并发上传事务，竞态仍存在"

        upload_txn.__exit__(None, None, None)  # 新上传先提交
        t.join(15)
        assert finished.is_set()

        assert "unexpected" not in outcome, outcome.get("unexpected")
        assert "published" not in outcome  # 未成功发布
        assert "不可发布" in outcome["error"]

        rows = _assert_no_residue(db_file)
        assert rows[stale_id]["status"] == db.STATUS_REJECTED
        assert rows[stale_id]["output_csv"] is None
        assert rows[stale_id]["output_filename"] is None
        assert rows[new_id]["status"] == db.STATUS_PENDING
    finally:
        if uploader.in_transaction:
            uploader.execute("ROLLBACK")
        uploader.close()


def test_publish_committed_first_survives_later_upload(db_file):
    """旧确认先提交：已发布内容保持可下载；后到的上传只影响 pending。"""
    with db.session(db_file) as conn:
        old_id = db.insert_pending(conn, "old.csv", 6, _plan_json())
        db.publish_batch(conn, old_id, "年龄,邮编,病种\n40,10000,糖尿病\n".encode(),
                         f"anonymous_batch_{old_id}.csv")
        new_id = db.insert_pending(conn, "new.csv", 6, _plan_json("C"))

    rows = _assert_no_residue(db_file)
    assert rows[old_id]["status"] == db.STATUS_PUBLISHED
    assert rows[old_id]["output_csv"] is not None  # 已发布文件保持可下载
    assert rows[new_id]["status"] == db.STATUS_PENDING  # 新批次等待确认


def test_publish_holding_lock_blocks_upload_until_commit(db_file):
    """旧确认持有写锁时，上传必须等待；提交后上传只接管待发布预览。"""
    with db.session(db_file) as conn:
        old_id = db.insert_pending(conn, "old.csv", 6, _plan_json())

    upload_done = threading.Event()
    upload_result = {}

    def attempt_upload():
        uploader = db.connect(db_file)  # SQLite 连接不可跨线程使用
        try:
            upload_result["id"] = db.insert_pending(
                uploader, "new.csv", 6, _plan_json("C"))
        finally:
            uploader.close()
            upload_done.set()

    with db.write_transaction(db.connect(db_file)) as pub:
        t = threading.Thread(target=attempt_upload)
        t.start()
        assert not upload_done.wait(0.3), "上传未等待发布事务提交"
        # 在持有的写锁内完成确认发布（与 publish_batch 相同的读取-写入顺序）
        row = pub.execute(
            "SELECT status FROM batches WHERE id = ?", (old_id,)).fetchone()
        assert row["status"] == db.STATUS_PENDING
        pub.execute(
            "UPDATE batches SET status = 'published', output_csv = ?, "
            "output_filename = ? WHERE id = ? AND status = 'pending'",
            (b"payload", f"anonymous_batch_{old_id}.csv", old_id))
    t.join(15)
    assert upload_done.is_set()

    rows = _assert_no_residue(db_file)
    assert rows[old_id]["status"] == db.STATUS_PUBLISHED
    assert rows[old_id]["output_csv"] == b"payload"
    new_id = upload_result["id"]
    assert rows[new_id]["status"] == db.STATUS_PENDING


def test_http_stale_preview_rejected_without_download(client):
    """HTTP 层：新上传先提交后，旧确认返回 409 且下载 404。"""
    c, _ = client
    c.post("/upload", files={"file": ("old.csv", _csv_bytes(VALID), "text/csv")})
    c.post("/upload", files={"file": ("new.csv", _csv_bytes(OTHER), "text/csv")})

    r = c.post("/publish/1", data={"confirm": "on"})
    assert r.status_code == 409
    assert "不可发布" in r.text
    assert c.get("/download/1").status_code == 404

    assert c.post("/publish/2", data={"confirm": "on"},
                  follow_redirects=False).status_code == 303
    assert c.get("/download/2").status_code == 200


def test_http_publish_committed_first_keeps_download(client):
    """HTTP 层：旧确认先提交后，已发布文件仍可下载，新批次等待确认。"""
    c, _ = client
    c.post("/upload", files={"file": ("old.csv", _csv_bytes(VALID), "text/csv")})
    assert c.post("/publish/1", data={"confirm": "on"},
                  follow_redirects=False).status_code == 303
    c.post("/upload", files={"file": ("new.csv", _csv_bytes(OTHER), "text/csv")})

    d = c.get("/download/1")
    assert d.status_code == 200 and d.content
    home = c.get("/").text
    assert "批次 #2" in home and "确认发布" in home
    assert "anonymous_batch_1.csv" in home


def test_failed_publish_leaves_no_downloadable_residue(client, monkeypatch):
    """发布前复核失败：返回 409，状态仍为 pending 且不写任何输出。"""
    c, db_file = client
    c.post("/upload", files={"file": ("d.csv", _csv_bytes(VALID), "text/csv")})

    from app import main as app_main
    monkeypatch.setattr(app_main.az, "verify_output", lambda _rows: False)
    r = c.post("/publish/1", data={"confirm": "on"})
    assert r.status_code == 409
    assert "复核" in r.text
    assert c.get("/download/1").status_code == 404

    with db.session(db_file) as conn:
        row = db.get_batch(conn, 1)
    assert row["status"] == db.STATUS_PENDING
    assert row["output_csv"] is None
    assert row["output_filename"] is None
