from __future__ import annotations
from .mixins import ImmutableDictMixin
from .mixins import UpdateDictMixin
from .. import http
def cache_control_property(key, empty, type):
    """Return a new property object for a cache header. Useful if you
    want to add support for a cache extension in a subclass.

    .. versionchanged:: 2.0
        Renamed from ``cache_property``.
    """
    return property(lambda x: x._get_cache_value(key, empty, type), lambda x, v: x._set_cache_value(key, v, type), lambda x: x._del_cache_value(key), f'accessor for {key!r}')