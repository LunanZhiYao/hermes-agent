#!/usr/bin/env python3
"""
Session state storage for Hermes — MySQL 8+ via SQLAlchemy Core (PyMySQL).

Fresh deploy only: no migrations from legacy SQLite. Schema version is
recorded in ``schema_version``; mismatch raises at startup.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    delete,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError

from hermes_cli.db_engine import get_engine

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 8

_FULLTEXT_INDEX_NAME = "ft_messages_content"
_DDL_RETRY_MYSQL_ERROR_CODES = frozenset({1050, 1205, 1213, 1684})


def _is_retryable_mysql_ddl_error(exc: Exception) -> bool:
    if not isinstance(exc, OperationalError):
        return False
    orig = getattr(exc, "orig", None)
    if orig is None or not hasattr(orig, "args") or not orig.args:
        return False
    try:
        code = int(orig.args[0])
    except (TypeError, ValueError):
        return False
    return code in _DDL_RETRY_MYSQL_ERROR_CODES


def _run_with_mysql_ddl_retry(fn, *, attempts: int = 6, delay_s: float = 0.2) -> None:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            fn()
            return
        except Exception as exc:
            if not _is_retryable_mysql_ddl_error(exc) or i == attempts - 1:
                raise
            last_exc = exc
            time.sleep(delay_s * (i + 1))
    if last_exc is not None:
        raise last_exc


def _require_mysql_engine(engine: Engine) -> None:
    if not str(engine.url).startswith("mysql"):
        raise TypeError("SessionDB requires a mysql+pymysql engine (no SQLite or other drivers)")


class SessionDB:
    """MySQL-backed session storage (SQLAlchemy Core) with FULLTEXT + LIKE fallback."""

    MAX_TITLE_LENGTH = 100

    def __init__(self, engine: Optional[Engine] = None) -> None:
        self._owns_engine = engine is None
        self._engine = get_engine() if engine is None else engine
        _require_mysql_engine(self._engine)
        self._lock = threading.Lock()
        self._metadata = MetaData()
        self._schema_version = Table(
            "schema_version",
            self._metadata,
            Column("version", Integer, primary_key=True, autoincrement=False),
        )
        self._sessions = Table(
            "sessions",
            self._metadata,
            Column("id", String(255), primary_key=True),
            Column("source", String(64), nullable=False),
            Column("user_id", String(255)),
            Column("model", String(255)),
            Column("model_config", Text),
            Column("system_prompt", Text),
            Column("parent_session_id", String(255), ForeignKey("sessions.id"), nullable=True),
            Column("started_at", Float, nullable=False),
            Column("ended_at", Float),
            Column("end_reason", String(64)),
            Column("message_count", Integer, nullable=False, default=0),
            Column("tool_call_count", Integer, nullable=False, default=0),
            Column("input_tokens", Integer, nullable=False, default=0),
            Column("output_tokens", Integer, nullable=False, default=0),
            Column("cache_read_tokens", Integer, nullable=False, default=0),
            Column("cache_write_tokens", Integer, nullable=False, default=0),
            Column("reasoning_tokens", Integer, nullable=False, default=0),
            Column("billing_provider", String(255)),
            Column("billing_base_url", String(1024)),
            Column("billing_mode", String(64)),
            Column("estimated_cost_usd", Float),
            Column("actual_cost_usd", Float),
            Column("cost_status", String(64)),
            Column("cost_source", String(64)),
            Column("pricing_version", String(64)),
            Column("title", String(255), unique=True, nullable=True),
            Column("api_call_count", Integer, nullable=False, default=0),
        )
        self._messages = Table(
            "messages",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("session_id", String(255), ForeignKey("sessions.id"), nullable=False),
            Column("role", String(32), nullable=False),
            Column("content", Text),
            Column("tool_call_id", String(255)),
            Column("tool_calls", Text),
            Column("tool_name", String(255)),
            Column("timestamp", Float, nullable=False),
            Column("token_count", Integer),
            Column("finish_reason", String(64)),
            Column("reasoning", Text),
            Column("reasoning_content", Text),
            Column("reasoning_details", Text),
            Column("codex_reasoning_items", Text),
        )
        self._state_meta = Table(
            "state_meta",
            self._metadata,
            Column("key", String(255), primary_key=True),
            Column("value", Text),
        )
        _run_with_mysql_ddl_retry(
            lambda: self._metadata.create_all(
                self._engine,
                tables=[self._schema_version, self._sessions, self._messages, self._state_meta],
            )
        )
        self._init_schema_version_row()
        self._ensure_fulltext_index()

    def _init_schema_version_row(self) -> None:
        with self._engine.begin() as conn:
            row = conn.execute(select(self._schema_version.c.version).limit(1)).first()
            if row is None:
                conn.execute(insert(self._schema_version).values(version=SCHEMA_VERSION))
            elif int(row[0]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema version is {row[0]}, expected {SCHEMA_VERSION}. "
                    "This build targets a single fresh MySQL install (no auto-migration)."
                )

    def _ensure_fulltext_index(self) -> None:
        with self._engine.begin() as conn:
            c = conn.execute(
                text(
                    "SELECT COUNT(1) FROM information_schema.statistics "
                    "WHERE table_schema = DATABASE() AND table_name = 'messages' "
                    f"AND index_name = :name"
                ),
                {"name": _FULLTEXT_INDEX_NAME},
            ).scalar()
            if int(c or 0) == 0:
                try:
                    conn.execute(
                        text(
                            f"CREATE FULLTEXT INDEX { _FULLTEXT_INDEX_NAME } "
                            "ON messages (content)"
                        )
                    )
                except (ProgrammingError, OperationalError) as exc:
                    logger.warning("Could not create FULLTEXT index on messages.content: %s", exc)

    def close(self) -> None:
        if self._owns_engine and self._engine is not None:
            self._engine.dispose()
        return None

    # =========================================================================
    # Session lifecycle
    # =========================================================================

    def create_session(
        self,
        session_id: str,
        source: str,
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        user_id: str = None,
        parent_session_id: str = None,
    ) -> str:
        payload = {
            "id": session_id,
            "source": source,
            "user_id": user_id,
            "model": model,
            "model_config": json.dumps(model_config) if model_config else None,
            "system_prompt": system_prompt,
            "parent_session_id": parent_session_id,
            "started_at": time.time(),
            "message_count": 0,
            "tool_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "api_call_count": 0,
        }
        with self._engine.begin() as conn:
            exists = conn.execute(
                select(self._sessions.c.id).where(self._sessions.c.id == session_id)
            ).first()
            if not exists:
                conn.execute(insert(self._sessions).values(**payload))
        return session_id

    def end_session(self, session_id: str, end_reason: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                update(self._sessions)
                .where(
                    and_(self._sessions.c.id == session_id, self._sessions.c.ended_at.is_(None))
                )
                .values(ended_at=time.time(), end_reason=end_reason)
            )

    def reopen_session(self, session_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                update(self._sessions)
                .where(self._sessions.c.id == session_id)
                .values(ended_at=None, end_reason=None)
            )

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                update(self._sessions)
                .where(self._sessions.c.id == session_id)
                .values(system_prompt=system_prompt)
            )

    def update_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
        api_call_count: int = 0,
        absolute: bool = False,
    ) -> None:
        with self._engine.begin() as conn:
            row = (
                conn.execute(select(self._sessions).where(self._sessions.c.id == session_id))
                .mappings()
                .first()
            )
            if not row:
                return
            values = {
                "input_tokens": input_tokens
                if absolute
                else int(row["input_tokens"] or 0) + input_tokens,
                "output_tokens": output_tokens
                if absolute
                else int(row["output_tokens"] or 0) + output_tokens,
                "cache_read_tokens": cache_read_tokens
                if absolute
                else int(row["cache_read_tokens"] or 0) + cache_read_tokens,
                "cache_write_tokens": cache_write_tokens
                if absolute
                else int(row["cache_write_tokens"] or 0) + cache_write_tokens,
                "reasoning_tokens": reasoning_tokens
                if absolute
                else int(row["reasoning_tokens"] or 0) + reasoning_tokens,
                "api_call_count": api_call_count
                if absolute
                else int(row["api_call_count"] or 0) + api_call_count,
            }
            if estimated_cost_usd is not None:
                values["estimated_cost_usd"] = (
                    estimated_cost_usd
                    if absolute
                    else float(row["estimated_cost_usd"] or 0) + estimated_cost_usd
                )
            if actual_cost_usd is not None:
                values["actual_cost_usd"] = (
                    actual_cost_usd
                    if absolute
                    else float(row["actual_cost_usd"] or 0) + actual_cost_usd
                )
            if cost_status is not None:
                values["cost_status"] = cost_status
            if cost_source is not None:
                values["cost_source"] = cost_source
            if pricing_version is not None:
                values["pricing_version"] = pricing_version
            if billing_provider is not None and row["billing_provider"] is None:
                values["billing_provider"] = billing_provider
            if billing_base_url is not None and row["billing_base_url"] is None:
                values["billing_base_url"] = billing_base_url
            if billing_mode is not None and row["billing_mode"] is None:
                values["billing_mode"] = billing_mode
            if model is not None and row["model"] is None:
                values["model"] = model
            conn.execute(
                update(self._sessions).where(self._sessions.c.id == session_id).values(**values)
            )

    def ensure_session(
        self,
        session_id: str,
        source: str = "unknown",
        model: str = None,
    ) -> None:
        with self._engine.begin() as conn:
            exists = conn.execute(
                select(self._sessions.c.id).where(self._sessions.c.id == session_id)
            ).first()
            if not exists:
                conn.execute(
                    insert(self._sessions).values(
                        id=session_id,
                        source=source,
                        model=model,
                        started_at=time.time(),
                        message_count=0,
                        tool_call_count=0,
                        input_tokens=0,
                        output_tokens=0,
                        cache_read_tokens=0,
                        cache_write_tokens=0,
                        reasoning_tokens=0,
                        api_call_count=0,
                    )
                )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._engine.connect() as conn:
            row = (
                conn.execute(select(self._sessions).where(self._sessions.c.id == session_id))
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        exact = self.get_session(session_id_or_prefix)
        if exact:
            return exact["id"]
        escaped = (
            session_id_or_prefix.replace("\\", "\\\\")
            .replace("%", r"\%")
            .replace("_", r"\_")
        )
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id FROM sessions "
                    "WHERE id LIKE :prefix ESCAPE '\\\\' "
                    "ORDER BY started_at DESC, id DESC "
                    "LIMIT 2"
                ),
                {"prefix": f"{escaped}%"},
            ).all()
        matches = [row[0] for row in rows]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        if not title:
            return None
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", title)
        cleaned = re.sub(
            r"[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]",
            "",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None
        if len(cleaned) > SessionDB.MAX_TITLE_LENGTH:
            raise ValueError(
                f"Title too long ({len(cleaned)} chars, max {SessionDB.MAX_TITLE_LENGTH})"
            )
        return cleaned

    def set_session_title(self, session_id: str, title: str) -> bool:
        title = self.sanitize_title(title)
        with self._engine.begin() as conn:
            if title:
                conflict = conn.execute(
                    select(self._sessions.c.id).where(
                        and_(self._sessions.c.title == title, self._sessions.c.id != session_id)
                    )
                ).first()
                if conflict:
                    raise ValueError(
                        f"Title '{title}' is already in use by session {conflict[0]}"
                    )
            result = conn.execute(
                update(self._sessions)
                .where(self._sessions.c.id == session_id)
                .values(title=title)
            )
            return result.rowcount > 0

    def get_session_title(self, session_id: str) -> Optional[str]:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(self._sessions.c.title).where(self._sessions.c.id == session_id)
            ).first()
        return row[0] if row else None

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        with self._engine.connect() as conn:
            row = (
                conn.execute(select(self._sessions).where(self._sessions.c.title == title))
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def resolve_session_by_title(self, title: str) -> Optional[str]:
        exact = self.get_session_by_title(title)
        escaped = title.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
        with self._engine.connect() as conn:
            numbered = (
                conn.execute(
                    text(
                        "SELECT id, title, started_at FROM sessions "
                        "WHERE title LIKE :p ESCAPE '\\\\' ORDER BY started_at DESC, id DESC"
                    ),
                    {"p": f"{escaped} #%"},
                )
                .mappings()
                .all()
            )
        if numbered:
            return numbered[0]["id"]
        if exact:
            return exact["id"]
        return None

    def get_next_title_in_lineage(self, base_title: str) -> str:
        m = re.match(r"^(.*?) #(\d+)$", base_title)
        base = m.group(1) if m else base_title
        escaped = base.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT title FROM sessions "
                        "WHERE title = :base OR (title LIKE :p ESCAPE '\\\\')"
                    ),
                    {"base": base, "p": f"{escaped} #%"},
                )
            ).all()
        existing = [r[0] for r in rows if r[0]]
        if not existing:
            return base
        max_num = 1
        for t in existing:
            mm = re.match(r"^.* #(\d+)$", t)
            if mm:
                max_num = max(max_num, int(mm.group(1)))
        return f"{base} #{max_num + 1}"

    def get_compression_tip(self, session_id: str) -> Optional[str]:
        current = session_id
        for _ in range(100):
            with self._engine.connect() as conn:
                row = (
                    conn.execute(
                        text(
                            "SELECT id FROM sessions "
                            "WHERE parent_session_id = :cur "
                            "  AND started_at >= ("
                            "      SELECT ended_at FROM sessions "
                            "      WHERE id = :cur2 AND end_reason = 'compression'"
                            "  ) "
                            "ORDER BY started_at DESC LIMIT 1"
                        ),
                        {"cur": current, "cur2": current},
                    )
                    .mappings()
                    .first()
                )
            if not row:
                return current
            current = row["id"]
        return current

    def list_sessions_rich(
        self,
        source: str = None,
        exclude_sources: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
        project_compression_tips: bool = True,
    ) -> List[Dict[str, Any]]:
        where_clauses: List[str] = []
        params: Dict[str, Any] = {"lim": limit, "off": offset}

        if not include_children:
            where_clauses.append("s.parent_session_id IS NULL")
        if source:
            where_clauses.append("s.source = :source")
            params["source"] = source
        if exclude_sources:
            ph = ", ".join(f":ex_{i}" for i in range(len(exclude_sources)))
            for i, s in enumerate(exclude_sources):
                params[f"ex_{i}"] = s
            where_clauses.append(f"s.source NOT IN ({ph})")
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        q = f"""
            SELECT s.*,
                COALESCE(
                    (SELECT SUBSTR(
                        REPLACE(REPLACE(m.content, CHAR(10), ' '), CHAR(13), ' '),
                        1, 63
                     )
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                COALESCE(
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                    s.started_at
                ) AS last_active
            FROM sessions s
            {where_sql}
            ORDER BY s.started_at DESC
            LIMIT :lim OFFSET :off
        """
        with self._engine.connect() as conn:
            rows = conn.execute(text(q), params).mappings().all()
        sessions: List[Dict[str, Any]] = []
        for row in rows:
            s = dict(row)
            raw = (s.pop("_preview_raw", "") or "").strip()
            if raw:
                text_prev = raw[:60]
                s["preview"] = text_prev + ("..." if len(raw) > 60 else "")
            else:
                s["preview"] = ""
            if (
                s.get("message_count", 0) > 0
                and s.get("last_active") is not None
                and s.get("started_at") is not None
                and float(s["last_active"]) <= float(s["started_at"])
            ):
                s["last_active"] = float(s["started_at"]) + 0.001
            sessions.append(s)

        if project_compression_tips and not include_children:
            projected: List[Dict[str, Any]] = []
            for s in sessions:
                if s.get("end_reason") != "compression":
                    projected.append(s)
                    continue
                tip_id = self.get_compression_tip(s["id"])
                if tip_id == s["id"]:
                    projected.append(s)
                    continue
                tip_row = self._get_session_rich_row(tip_id)
                if not tip_row:
                    projected.append(s)
                    continue
                merged = dict(s)
                for key in (
                    "id",
                    "ended_at",
                    "end_reason",
                    "message_count",
                    "tool_call_count",
                    "title",
                    "last_active",
                    "preview",
                    "model",
                    "system_prompt",
                ):
                    if key in tip_row:
                        merged[key] = tip_row[key]
                merged["_lineage_root_id"] = s["id"]
                projected.append(merged)
            sessions = projected
        return sessions

    def _get_session_rich_row(self, session_id: str) -> Optional[Dict[str, Any]]:
        q = """
            SELECT s.*,
                COALESCE(
                    (SELECT SUBSTR(
                        REPLACE(REPLACE(m.content, CHAR(10), ' '), CHAR(13), ' '),
                        1, 63
                     )
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                COALESCE(
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                    s.started_at
                ) AS last_active
            FROM sessions s
            WHERE s.id = :sid
        """
        with self._engine.connect() as conn:
            row = conn.execute(text(q), {"sid": session_id}).mappings().first()
        if not row:
            return None
        s = dict(row)
        raw = (s.pop("_preview_raw", "") or "").strip()
        if raw:
            tprev = raw[:60]
            s["preview"] = tprev + ("..." if len(raw) > 60 else "")
        else:
            s["preview"] = ""
        if (
            s.get("message_count", 0) > 0
            and s.get("last_active") is not None
            and s.get("started_at") is not None
            and float(s["last_active"]) <= float(s["started_at"])
        ):
            s["last_active"] = float(s["started_at"]) + 0.001
        return s

    # =========================================================================
    # Message storage
    # =========================================================================

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning: str = None,
        reasoning_content: str = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
    ) -> int:
        reasoning_details_json = json.dumps(reasoning_details) if reasoning_details else None
        codex_items_json = json.dumps(codex_reasoning_items) if codex_reasoning_items else None
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        num_tool_calls = 0
        if tool_calls is not None:
            num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else 1
        with self._engine.begin() as conn:
            r = conn.execute(
                insert(self._messages).values(
                    session_id=session_id,
                    role=role,
                    content=content,
                    tool_call_id=tool_call_id,
                    tool_calls=tool_calls_json,
                    tool_name=tool_name,
                    timestamp=time.time(),
                    token_count=token_count,
                    finish_reason=finish_reason,
                    reasoning=reasoning,
                    reasoning_content=reasoning_content,
                    reasoning_details=reasoning_details_json,
                    codex_reasoning_items=codex_items_json,
                )
            )
            last_id = r.inserted_primary_key[0] if r.inserted_primary_key else None
            if num_tool_calls > 0:
                conn.execute(
                    update(self._sessions)
                    .where(self._sessions.c.id == session_id)
                    .values(
                        message_count=self._sessions.c.message_count + 1,
                        tool_call_count=self._sessions.c.tool_call_count + num_tool_calls,
                    )
                )
            else:
                conn.execute(
                    update(self._sessions)
                    .where(self._sessions.c.id == session_id)
                    .values(message_count=self._sessions.c.message_count + 1)
                )
        return int(last_id) if last_id is not None else 0

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(self._messages)
                    .where(self._messages.c.session_id == session_id)
                    .order_by(self._messages.c.timestamp.asc(), self._messages.c.id.asc())
                )
                .mappings()
                .all()
            )
        result: List[Dict[str, Any]] = []
        for row in rows:
            msg = dict(row)
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in get_messages, falling back to []")
                    msg["tool_calls"] = []
            result.append(msg)
        return result

    def get_messages_as_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(
                        self._messages.c.role,
                        self._messages.c.content,
                        self._messages.c.tool_call_id,
                        self._messages.c.tool_calls,
                        self._messages.c.tool_name,
                        self._messages.c.reasoning,
                        self._messages.c.reasoning_content,
                        self._messages.c.reasoning_details,
                        self._messages.c.codex_reasoning_items,
                    )
                    .where(self._messages.c.session_id == session_id)
                    .order_by(self._messages.c.timestamp.asc(), self._messages.c.id.asc())
                )
                .mappings()
                .all()
            )
        messages: List[Dict[str, Any]] = []
        for row in rows:
            msg: Dict[str, Any] = {"role": row["role"], "content": row["content"]}
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Failed to deserialize tool_calls in conversation replay, falling back to []"
                    )
                    msg["tool_calls"] = []
            if row["role"] == "assistant":
                if row["reasoning"]:
                    msg["reasoning"] = row["reasoning"]
                if row["reasoning_content"] is not None:
                    msg["reasoning_content"] = row["reasoning_content"]
                if row["reasoning_details"]:
                    try:
                        msg["reasoning_details"] = json.loads(row["reasoning_details"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(
                            "Failed to deserialize reasoning_details, falling back to None"
                        )
                        msg["reasoning_details"] = None
                if row["codex_reasoning_items"]:
                    try:
                        msg["codex_reasoning_items"] = json.loads(row["codex_reasoning_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(
                            "Failed to deserialize codex_reasoning_items, falling back to None"
                        )
                        msg["codex_reasoning_items"] = None
            messages.append(msg)
        return messages

    # =========================================================================
    # Search
    # =========================================================================

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        _quoted_parts: list = []

        def _preserve_quoted(m: re.Match) -> str:
            _quoted_parts.append(m.group(0))
            return f"\x00Q{len(_quoted_parts) - 1}\x00"

        sanitized = re.sub(r'"[^"]*"', _preserve_quoted, query)
        sanitized = re.sub(r'[+{}()\"^]', " ", sanitized)
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)
        sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
        sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())
        sanitized = re.sub(r"\b(\w+(?:[.-]\w+)+)\b", r'"\1"', sanitized)
        for i, quoted in enumerate(_quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)
        return sanitized.strip()

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        for ch in text:
            cp = ord(ch)
            if (
                0x4E00 <= cp <= 0x9FFF
                or 0x3400 <= cp <= 0x4DBF
                or 0x20000 <= cp <= 0x2A6DF
                or 0x3000 <= cp <= 0x303F
                or 0x3040 <= cp <= 0x309F
                or 0x30A0 <= cp <= 0x30FF
                or 0xAC00 <= cp <= 0xD7AF
            ):
                return True
        return False

    def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        sanitized = self._sanitize_fts5_query(query)
        if not sanitized:
            return []
        q_nl = re.sub(r"\s+", " ", re.sub(r'"', " ", sanitized)).strip()

        matches: List[Dict[str, Any]] = []
        # Try natural language, then boolean (quoted phrases), then CJK/ LIKE fallbacks
        for mode in ("NL", "BOOL"):
            q = q_nl if mode == "NL" else sanitized
            if mode == "NL" and not q_nl:
                continue
            if mode == "BOOL" and not sanitized:
                continue
            w_parts = ["m.content IS NOT NULL"]
            if mode == "NL":
                w_parts.append(
                    "MATCH (m.content) AGAINST (:ftq IN NATURAL LANGUAGE MODE)"
                )
                rank = "MATCH (m.content) AGAINST (:ftq IN NATURAL LANGUAGE MODE) DESC, m.timestamp DESC"
            else:
                w_parts.append("MATCH (m.content) AGAINST (:ftq IN BOOLEAN MODE)")
                rank = "MATCH (m.content) AGAINST (:ftq IN BOOLEAN MODE) DESC, m.timestamp DESC"
            p: Dict[str, Any] = {"ftq": q, "lim": limit, "off": offset}
            if source_filter is not None:
                ph = ", ".join(f":sf_{i}" for i in range(len(source_filter)))
                w_parts.append(f"s.source IN ({ph})")
                for i, v in enumerate(source_filter):
                    p[f"sf_{i}"] = v
            if exclude_sources is not None:
                ph = ", ".join(f":xs_{i}" for i in range(len(exclude_sources)))
                w_parts.append(f"s.source NOT IN ({ph})")
                for i, v in enumerate(exclude_sources):
                    p[f"xs_{i}"] = v
            if role_filter:
                ph = ", ".join(f":rf_{i}" for i in range(len(role_filter)))
                w_parts.append(f"m.role IN ({ph})")
                for i, v in enumerate(role_filter):
                    p[f"rf_{i}"] = v
            wh = " AND ".join(w_parts)
            sql = f"""
            SELECT
                m.id,
                m.session_id,
                m.role,
                m.content,
                m.timestamp,
                m.tool_name,
                s.source,
                s.model,
                s.started_at AS session_started
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE {wh}
            ORDER BY {rank}
            LIMIT :lim OFFSET :off
            """
            try:
                with self._engine.connect() as conn:
                    rows = conn.execute(text(sql), p).mappings().all()
                matches = [dict(x) for x in rows]
            except (ProgrammingError, OperationalError):
                matches = []
            if matches:
                break

        cjk = self._contains_cjk(sanitized)
        if not matches and cjk:
            raw_q = sanitized.strip('"').strip()
            w_parts2 = ["m.content LIKE :likeq"]
            p2: Dict[str, Any] = {
                "likeq": f"%{raw_q}%",
                "instr": raw_q,
                "lim": limit,
                "off": offset,
            }
            if source_filter is not None:
                ph = ", ".join(f":sf_{i}" for i in range(len(source_filter)))
                w_parts2.append(f"s.source IN ({ph})")
                for i, v in enumerate(source_filter):
                    p2[f"sf_{i}"] = v
            if exclude_sources is not None:
                ph = ", ".join(f":xs_{i}" for i in range(len(exclude_sources)))
                w_parts2.append(f"s.source NOT IN ({ph})")
                for i, v in enumerate(exclude_sources):
                    p2[f"xs_{i}"] = v
            if role_filter:
                ph = ", ".join(f":rf_{i}" for i in range(len(role_filter)))
                w_parts2.append(f"m.role IN ({ph})")
                for i, v in enumerate(role_filter):
                    p2[f"rf_{i}"] = v
            wh2 = " AND ".join(w_parts2)
            sql_like = f"""
                SELECT m.id, m.session_id, m.role, m.content,
                    SUBSTRING(m.content,
                        GREATEST(1, LOCATE(:instr, m.content) - 40), 120) AS snippet,
                    m.timestamp, m.tool_name, s.source, s.model, s.started_at AS session_started
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE {wh2}
                ORDER BY m.timestamp DESC
                LIMIT :lim OFFSET :off
            """
            with self._engine.connect() as conn:
                matches = [dict(x) for x in conn.execute(text(sql_like), p2).mappings().all()]

        if not matches and not cjk and q_nl:
            likeq = f"%{q_nl}%"
            w_parts3 = ["m.content IS NOT NULL", "m.content LIKE :likebroad"]
            p3: Dict[str, Any] = {
                "likebroad": likeq,
                "lim": limit,
                "off": offset,
            }
            if source_filter is not None:
                ph = ", ".join(f":sf_{i}" for i in range(len(source_filter)))
                w_parts3.append(f"s.source IN ({ph})")
                for i, v in enumerate(source_filter):
                    p3[f"sf_{i}"] = v
            if exclude_sources is not None:
                ph = ", ".join(f":xs_{i}" for i in range(len(exclude_sources)))
                w_parts3.append(f"s.source NOT IN ({ph})")
                for i, v in enumerate(exclude_sources):
                    p3[f"xs_{i}"] = v
            if role_filter:
                ph = ", ".join(f":rf_{i}" for i in range(len(role_filter)))
                w_parts3.append(f"m.role IN ({ph})")
                for i, v in enumerate(role_filter):
                    p3[f"rf_{i}"] = v
            wh3 = " AND ".join(w_parts3)
            sql_broad = f"""
                SELECT m.id, m.session_id, m.role, m.content,
                    SUBSTRING(m.content, 1, 120) AS snippet,
                    m.timestamp, m.tool_name, s.source, s.model, s.started_at AS session_started
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE {wh3}
                ORDER BY m.timestamp DESC
                LIMIT :lim OFFSET :off
            """
            with self._engine.connect() as conn:
                matches = [dict(x) for x in conn.execute(text(sql_broad), p3).mappings().all()]

        for mrow in matches:
            content = mrow.get("content") or ""
            if "snippet" not in mrow or mrow["snippet"] is None:
                n = q_nl or sanitized.replace('"', "")
                n = n.split()[0] if n.split() else n
                pos = content.lower().find(n.lower()[:20]) if n else -1
                if pos >= 0:
                    start = max(0, pos - 40)
                    mrow["snippet"] = content[start : start + 120]
                else:
                    mrow["snippet"] = content[:120]
            try:
                with self._engine.connect() as conn:
                    ctx = conn.execute(
                        text(
                            """WITH target AS (
                               SELECT session_id, timestamp, id
                               FROM messages
                               WHERE id = :mid
                           )
                           SELECT role, content
                           FROM (
                               SELECT m.id, m.timestamp, m.role, m.content
                               FROM messages m
                               JOIN target t ON t.session_id = m.session_id
                               WHERE (m.timestamp < t.timestamp)
                                  OR (m.timestamp = t.timestamp AND m.id < t.id)
                               ORDER BY m.timestamp DESC, m.id DESC
                               LIMIT 1
                           ) AS prev
                           UNION ALL
                           SELECT role, content
                           FROM messages
                           WHERE id = :mid2
                           UNION ALL
                           SELECT role, content
                           FROM (
                               SELECT m.id, m.timestamp, m.role, m.content
                               FROM messages m
                               JOIN target t ON t.session_id = m.session_id
                               WHERE (m.timestamp > t.timestamp)
                                  OR (m.timestamp = t.timestamp AND m.id > t.id)
                               ORDER BY m.timestamp ASC, m.id ASC
                               LIMIT 1
                           ) AS nxt """
                        ),
                        {"mid": mrow["id"], "mid2": mrow["id"]},
                    )
                    context_msgs = [
                        {"role": r["role"], "content": (r["content"] or "")[:200]} for r in ctx.mappings()
                    ]
            except (ProgrammingError, OperationalError):
                context_msgs = []
            mrow["context"] = context_msgs
            mrow.pop("content", None)
        return matches

    def search_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        stmt = select(self._sessions).order_by(
            self._sessions.c.started_at.desc()
        ).limit(limit).offset(offset)
        if source:
            stmt = stmt.where(self._sessions.c.source == source)
        with self._engine.connect() as conn:
            return [dict(x) for x in conn.execute(stmt).mappings().all()]

    # =========================================================================
    # Utility
    # =========================================================================

    def session_count(self, source: str = None) -> int:
        stmt = select(func.count()).select_from(self._sessions)
        if source:
            stmt = stmt.where(self._sessions.c.source == source)
        with self._engine.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def message_count(self, session_id: str = None) -> int:
        stmt = select(func.count()).select_from(self._messages)
        if session_id:
            stmt = stmt.where(self._messages.c.session_id == session_id)
        with self._engine.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            return None
        return {**session, "messages": self.get_messages(session_id)}

    def export_all(self, source: str = None) -> List[Dict[str, Any]]:
        sessions = self.search_sessions(source=source, limit=100000)
        return [{**s, "messages": self.get_messages(s["id"])} for s in sessions]

    def clear_messages(self, session_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._messages).where(self._messages.c.session_id == session_id)
            )
            conn.execute(
                update(self._sessions)
                .where(self._sessions.c.id == session_id)
                .values(message_count=0, tool_call_count=0)
            )

    def delete_session(self, session_id: str) -> bool:
        with self._engine.begin() as conn:
            exists = conn.execute(
                select(self._sessions.c.id).where(self._sessions.c.id == session_id)
            ).first()
            if not exists:
                return False
            conn.execute(
                update(self._sessions)
                .where(self._sessions.c.parent_session_id == session_id)
                .values(parent_session_id=None)
            )
            conn.execute(
                delete(self._messages).where(self._messages.c.session_id == session_id)
            )
            conn.execute(delete(self._sessions).where(self._sessions.c.id == session_id))
        return True

    def prune_sessions(self, older_than_days: int = 90, source: str = None) -> int:
        cutoff = time.time() - (older_than_days * 86400)
        with self._engine.begin() as conn:
            stmt = select(self._sessions.c.id).where(
                and_(
                    self._sessions.c.started_at < cutoff,
                    self._sessions.c.ended_at.isnot(None),
                )
            )
            if source:
                stmt = stmt.where(self._sessions.c.source == source)
            ids = [r[0] for r in conn.execute(stmt).all()]
            if not ids:
                return 0
            conn.execute(
                update(self._sessions)
                .where(self._sessions.c.parent_session_id.in_(ids))
                .values(parent_session_id=None)
            )
            conn.execute(
                delete(self._messages).where(self._messages.c.session_id.in_(ids))
            )
            conn.execute(
                delete(self._sessions).where(self._sessions.c.id.in_(ids))
            )
            return len(ids)

    def get_meta(self, key: str) -> Optional[str]:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(self._state_meta.c.value).where(self._state_meta.c.key == key)
            ).first()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._engine.begin() as conn:
            exists = conn.execute(
                select(self._state_meta.c.key).where(self._state_meta.c.key == key)
            ).first()
            if exists:
                conn.execute(
                    update(self._state_meta)
                    .where(self._state_meta.c.key == key)
                    .values(value=value)
                )
            else:
                conn.execute(insert(self._state_meta).values(key=key, value=value))

    def vacuum(self) -> None:
        return None

    def maybe_auto_prune_and_vacuum(
        self,
        retention_days: int = 90,
        min_interval_hours: int = 24,
        vacuum: bool = True,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"skipped": False, "pruned": 0, "vacuumed": False}
        try:
            last_raw = self.get_meta("last_auto_prune")
            now = time.time()
            if last_raw:
                try:
                    last_ts = float(last_raw)
                    if now - last_ts < min_interval_hours * 3600:
                        result["skipped"] = True
                        return result
                except (TypeError, ValueError):
                    pass
            pruned = self.prune_sessions(older_than_days=retention_days)
            result["pruned"] = pruned
            if vacuum and pruned > 0:
                self.vacuum()
                result["vacuumed"] = True
            self.set_meta("last_auto_prune", str(now))
            if pruned > 0:
                logger.info(
                    "MySQL state auto-maintenance: pruned %d session(s) older than %d days",
                    pruned,
                    retention_days,
                )
        except Exception as exc:
            logger.warning("MySQL state auto-maintenance failed: %s", exc)
            result["error"] = str(exc)
        return result
