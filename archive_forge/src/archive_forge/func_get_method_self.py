import inspect
import logging
import operator
import types
def get_method_self(method):
    """Gets the ``self`` object attached to this method (or none)."""
    if not inspect.ismethod(method):
        return None
    try:
        return operator.attrgetter('__self__')(method)
    except AttributeError:
        return None