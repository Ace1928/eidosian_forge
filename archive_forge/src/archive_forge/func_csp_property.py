from __future__ import annotations
from .mixins import UpdateDictMixin
def csp_property(key):
    """Return a new property object for a content security policy header.
    Useful if you want to add support for a csp extension in a
    subclass.
    """
    return property(lambda x: x._get_value(key), lambda x, v: x._set_value(key, v), lambda x: x._del_value(key), f'accessor for {key!r}')