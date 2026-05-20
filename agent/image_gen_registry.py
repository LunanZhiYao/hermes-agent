"""
Image Generation Provider Registry
==================================

Central map of registered providers. Populated by plugins at import-time via
``PluginContext.register_image_gen_provider()``; consumed by the
``image_generate`` tool to dispatch each call to the active backend.

Active selection
----------------
The active provider is chosen by ``image_gen.provider`` in ``config.yaml``.
If unset, :func:`get_active_provider` applies fallback logic:

1. If exactly one provider is registered, use it.
2. Otherwise if a provider named ``fal`` is registered, use it (legacy
   default — matches pre-plugin behavior).
3. Otherwise return ``None`` (the tool surfaces a helpful error pointing
   the user at ``hermes tools``).

Multi-tenant Support
--------------------
In SaaS multi-tenant mode, each tenant has its own HERMES_HOME directory
with potentially different plugins. The registry maintains a per-tenant
cache to avoid re-discovering plugins on every request while ensuring
isolation between tenants.

Use :func:`switch_tenant_context` to switch the active tenant context.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional

from agent.image_gen_provider import ImageGenProvider

logger = logging.getLogger(__name__)

# Current active providers (global or current tenant)
_providers: Dict[str, ImageGenProvider] = {}

# Per-tenant cache: {hermes_home_str: {provider_name: provider}}
_tenant_caches: Dict[str, Dict[str, ImageGenProvider]] = {}

# Track which HERMES_HOME the current _providers corresponds to
_current_hermes_home: Optional[str] = None

_lock = threading.Lock()


def _get_hermes_home_key() -> str:
    """Get the current HERMES_HOME as a cache key."""
    return os.environ.get("HERMES_HOME", "").strip() or "__default__"


def switch_tenant_context() -> bool:
    """Switch to the appropriate tenant context based on current HERMES_HOME.
    
    Returns True if context was switched (cache hit), False if no cache exists
    (caller should trigger plugin discovery).
    
    This is the core of multi-tenant support:
    - If HERMES_HOME matches current context, do nothing (fast path)
    - If HERMES_HOME changed and we have a cache, restore from cache
    - If HERMES_HOME changed and no cache, return False to signal discovery needed
    """
    global _providers, _current_hermes_home
    
    hermes_home = _get_hermes_home_key()
    
    with _lock:
        # Fast path: same context, nothing to do
        if hermes_home == _current_hermes_home:
            return True
        
        # Save current context before switching (if not empty)
        if _current_hermes_home is not None and _providers:
            _tenant_caches[_current_hermes_home] = dict(_providers)
            logger.debug(
                "Saved %d provider(s) to cache for HERMES_HOME=%s",
                len(_providers), _current_hermes_home
            )
        
        # Check if we have a cached context for the new HERMES_HOME
        cached = _tenant_caches.get(hermes_home)
        if cached is not None:
            _providers = dict(cached)
            _current_hermes_home = hermes_home
            logger.debug(
                "Restored %d provider(s) from cache for HERMES_HOME=%s",
                len(_providers), hermes_home
            )
            return True
        
        # No cache exists - clear and signal that discovery is needed
        _providers.clear()
        _current_hermes_home = hermes_home
        logger.debug("No cache for HERMES_HOME=%s, discovery needed", hermes_home)
        return False


def register_provider(provider: ImageGenProvider) -> None:
    """Register an image generation provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — this makes hot-reload scenarios (tests, dev loops)
    behave predictably.
    """
    if not isinstance(provider, ImageGenProvider):
        raise TypeError(
            f"register_provider() expects an ImageGenProvider instance, "
            f"got {type(provider).__name__}"
        )
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Image gen provider .name must be a non-empty string")
    
    hermes_home = _get_hermes_home_key()
    
    with _lock:
        # Ensure _current_hermes_home is set
        global _current_hermes_home
        if _current_hermes_home is None:
            _current_hermes_home = hermes_home
        
        existing = _providers.get(name)
        _providers[name] = provider
        
        # Also update the tenant cache
        if _current_hermes_home not in _tenant_caches:
            _tenant_caches[_current_hermes_home] = {}
        _tenant_caches[_current_hermes_home][name] = provider
    
    if existing is not None:
        logger.debug("Image gen provider '%s' re-registered (was %r)", name, type(existing).__name__)
    else:
        logger.debug("Registered image gen provider '%s' (%s) for HERMES_HOME=%s", 
                     name, type(provider).__name__, _current_hermes_home)


def list_providers() -> List[ImageGenProvider]:
    """Return all registered providers, sorted by name."""
    with _lock:
        items = list(_providers.values())
    return sorted(items, key=lambda p: p.name)


def get_provider(name: str) -> Optional[ImageGenProvider]:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None
    with _lock:
        return _providers.get(name.strip())


def get_active_provider() -> Optional[ImageGenProvider]:
    """Resolve the currently-active provider.

    Reads ``image_gen.provider`` from config.yaml; falls back per the
    module docstring.
    """
    configured: Optional[str] = None
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            raw = section.get("provider")
            if isinstance(raw, str) and raw.strip():
                configured = raw.strip()
    except Exception as exc:
        logger.debug("Could not read image_gen.provider from config: %s", exc)

    with _lock:
        snapshot = dict(_providers)

    if configured:
        provider = snapshot.get(configured)
        if provider is not None:
            return provider
        logger.debug(
            "image_gen.provider='%s' configured but not registered; falling back",
            configured,
        )

    # Fallback: single-provider case
    if len(snapshot) == 1:
        return next(iter(snapshot.values()))

    # Fallback: prefer legacy FAL for backward compat
    if "fal" in snapshot:
        return snapshot["fal"]

    return None


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    global _providers, _current_hermes_home
    with _lock:
        _providers.clear()
        _tenant_caches.clear()
        _current_hermes_home = None


def reset_registry() -> None:
    """Clear the current registry for plugin rediscovery.
    
    Note: This only clears the current tenant's providers, not the cache.
    Use :func:`switch_tenant_context` for proper multi-tenant handling.
    """
    with _lock:
        _providers.clear()
    logger.debug("Image gen provider registry cleared for rediscovery")


def get_cache_stats() -> Dict[str, int]:
    """Return cache statistics for debugging/monitoring."""
    with _lock:
        return {
            "current_providers": len(_providers),
            "cached_tenants": len(_tenant_caches),
            "current_hermes_home": _current_hermes_home or "(none)",
        }
