"""SQLite 持久化：批次（预览/发布）状态管理。

状态机
------
pending  : 已上传，存在可发布的匿名方案，等待确认发布
published: 已确认发布（含可下载 CSV），重启后内容不变
rejected : 已上传但无可行匿名方案，或预览已被新的上传取代

并发裁决
--------
上传（使旧 pending 失效 + 插入新批次）与确认发布都会先执行
``BEGIN IMMEDIATE`` 取得 SQLite 写锁，再在锁内读取状态并写入。
因此两者严格按数据库提交顺序串行化：

* 上传先提交 -> 旧确认在锁内读到 rejected，拒绝发布，不写任何输出；
* 确认先提交 -> 已发布批次保持 published 与可下载，随后提交的上传
  只会使仍处于 pending 的预览失效，新批次等待人工确认。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

STATUS_PENDING = "pending"
STATUS_PUBLISHED = "published"
STATUS_REJECTED = "rejected"

BUSY_TIMEOUT_MS = 30_000


class PreviewInvalidError(RuntimeError):
    """预览已不可发布：已被发布或已被新的上传取代。"""


class OutputVerificationError(RuntimeError):
    """匿名结果未通过发布前复核。"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    status          TEXT NOT NULL CHECK (status IN ('pending','published','rejected')),
    source_filename TEXT NOT NULL,
    row_count       INTEGER NOT NULL,
    plan_json       TEXT,
    output_csv      BLOB,
    output_filename TEXT
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """打开连接：自动提交模式，事务由本模块显式控制。"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        timeout=BUSY_TIMEOUT_MS / 1000,
        isolation_level=None,  # 关闭隐式事务，避免 SELECT 落在事务之外
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """请求级连接：退出时关闭连接（不自动提交任何未显式提交的事务）。"""
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _safe_rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ROLLBACK")
    except sqlite3.Error:
        pass  # 事务可能已被 SQLite 自动回滚


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """写事务：立即取得写锁（BEGIN IMMEDIATE），保证读到的状态即裁决依据。

    普通的 ``BEGIN``（延迟事务）在 SELECT 之后才尝试升级写锁，会让并发的
    上传在“读状态”与“写状态”之间提交，从而把已失效的旧预览改成已发布。
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        _safe_rollback(conn)
        raise
    else:
        try:
            conn.execute("COMMIT")
        except BaseException:
            _safe_rollback(conn)
            raise


@contextmanager
def read_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """只读快照事务：页面渲染时 pending 与 published 取自同一数据库快照。"""
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        _safe_rollback(conn)
        raise
    else:
        conn.execute("COMMIT")


def init_db(db_path: str | Path) -> None:
    with session(db_path) as conn:
        # WAL 记录模式对数据库文件持久生效：写入者互斥、读者不被写入阻塞，
        # BEGIN IMMEDIATE 的排队顺序即提交顺序。
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA)


def insert_pending(conn: sqlite3.Connection, filename: str, row_count: int,
                   plan_json: str) -> int:
    """单事务：使旧 pending 预览失效，再插入新批次（再次上传后旧预览不得发布）。"""
    with write_transaction(conn):
        conn.execute(
            "UPDATE batches SET status = 'rejected' WHERE status = 'pending'")
        cur = conn.execute(
            "INSERT INTO batches (status, source_filename, row_count, plan_json) "
            "VALUES ('pending', ?, ?, ?)",
            (filename, row_count, plan_json),
        )
        return int(cur.lastrowid)


def insert_rejected(conn: sqlite3.Connection, filename: str, row_count: int) -> int:
    with write_transaction(conn):
        conn.execute(
            "UPDATE batches SET status = 'rejected' WHERE status = 'pending'")
        cur = conn.execute(
            "INSERT INTO batches (status, source_filename, row_count) "
            "VALUES ('rejected', ?, ?)",
            (filename, row_count),
        )
        return int(cur.lastrowid)


def get_batch(conn: sqlite3.Connection, batch_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()


def get_latest_pending(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM batches WHERE status = 'pending' ORDER BY id DESC LIMIT 1"
    ).fetchone()


def list_published(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute(
        "SELECT * FROM batches WHERE status = 'published' ORDER BY id DESC"))


def publish_batch(conn: sqlite3.Connection, batch_id: int,
                  output_csv: bytes, output_filename: str) -> None:
    """原子发布：仅当前 pending 批次可发布；失败抛异常并整体回滚，不留残件。

    状态读取与状态写入都在 ``BEGIN IMMEDIATE`` 写锁内完成，上传无法在
    两者之间提交；若上传先提交，此处读到 rejected 并拒绝发布。
    """
    with write_transaction(conn):
        row = conn.execute(
            "SELECT status FROM batches WHERE id = ?", (batch_id,)).fetchone()
        if row is None:
            raise LookupError("批次不存在")
        if row["status"] != STATUS_PENDING:
            if row["status"] == STATUS_PUBLISHED:
                raise PreviewInvalidError("该批次已发布，不可重复发布")
            raise PreviewInvalidError(
                "该预览已失效，不可发布：已被新的上传取代")
        cur = conn.execute(
            "UPDATE batches SET status = 'published', output_csv = ?, "
            "output_filename = ? WHERE id = ? AND status = 'pending'",
            (output_csv, output_filename, batch_id),
        )
        if cur.rowcount != 1:  # 纵深防御：并发状态变更绝不允许写入输出
            raise PreviewInvalidError(
                "该预览已失效，不可发布：已被新的上传取代")
