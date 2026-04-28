# Session Storage

Hermes Agent uses a MySQL backend (via SQLAlchemy Core + PyMySQL) for
session metadata and message history across CLI and gateway runtime paths.

Source file: `hermes_state.py`

## Architecture Overview

Runtime storage tables:

- `sessions` ！ session metadata, counters, costs, titles, lineage
- `messages` ！ per-session transcript rows
- `state_meta` ！ key/value metadata used by maintenance tasks
- `schema_version` ！ expected schema marker (`8`)

Search index:

- `messages.content` has a MySQL `FULLTEXT` index
- query path uses `MATCH ... AGAINST`
- CJK and special cases fall back to `LIKE`

## Runtime Contract

- Session storage is **env-only** at runtime:
  - `DB_HOST`
  - `DB_PORT`
  - `DB_NAME`
  - `DB_USER`
  - `DB_PASSWORD`
- Optional:
  - `DB_CHARSET` (default: `utf8mb4`)
- Database expectations:
  - MySQL 8.0.x
  - charset `utf8mb4`
  - collation `utf8mb4_0900_ai_ci`

## SessionDB API

Core lifecycle:

- `create_session(...)`
- `end_session(...)`
- `reopen_session(...)`
- `ensure_session(...)`

Message I/O:

- `append_message(...)`
- `get_messages(...)`
- `get_messages_as_conversation(...)`
- `clear_messages(session_id)`

Search and listing:

- `search_messages(...)`
- `search_sessions(...)`
- `list_sessions_rich(...)`

Title and lineage:

- `set_session_title(...)`
- `get_session_title(...)`
- `resolve_session_by_title(...)`
- `get_next_title_in_lineage(...)`
- `get_compression_tip(...)`

Maintenance:

- `prune_sessions(...)`
- `maybe_auto_prune_and_vacuum(...)`
- `vacuum()` (no-op on MySQL build)

## Notes

- Fresh-deploy model: no SQLite migration path in runtime code.
- Gateway still maintains JSONL transcript files for backward compatibility
  when reading or merging legacy conversation history.