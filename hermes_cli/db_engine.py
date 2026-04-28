"""Shared SQLAlchemy engine for MySQL storage backends."""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote_plus

from sqlalchemy import Engine, create_engine

from hermes_cli.config import get_database_env_config

_ENGINE: Optional[Engine] = None


def get_engine() -> Engine:
    """Create (or return cached) SQLAlchemy engine from strict DB env vars."""
    global _ENGINE
    if _ENGINE is None:
        cfg = get_database_env_config()
        user = quote_plus(cfg["user"])
        password = quote_plus(cfg["password"])
        host = cfg["host"]
        port = cfg["port"]
        name = cfg["name"]
        charset = cfg["charset"]
        url = (
            f"mysql+pymysql://{user}:{password}"
            f"@{host}:{port}/{name}?charset={charset}"
        )
        _ENGINE = create_engine(
            url,
            pool_pre_ping=True,
            future=True,
        )
    return _ENGINE

