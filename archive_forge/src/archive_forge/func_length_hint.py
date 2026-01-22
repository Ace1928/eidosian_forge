from builtins import abs as _abs
def length_hint(obj, default=0):
    """
    Return an estimate of the number of items in obj.
    This is useful for presizing containers when building from an iterable.

    If the object supports len(), the result will be exact. Otherwise, it may
    over- or under-estimate by an arbitrary amount. The result will be an
    integer >= 0.
    """
    if not isinstance(default, int):
        msg = "'%s' object cannot be interpreted as an integer" % type(default).__name__
        raise TypeError(msg)
    try:
        return len(obj)
    except TypeError:
        pass
    try:
        hint = type(obj).__length_hint__
    except AttributeError:
        return default
    try:
        val = hint(obj)
    except TypeError:
        return default
    if val is NotImplemented:
        return default
    if not isinstance(val, int):
        msg = '__length_hint__ must be integer, not %s' % type(val).__name__
        raise TypeError(msg)
    if val < 0:
        msg = '__length_hint__() should return >= 0'
        raise ValueError(msg)
    return val