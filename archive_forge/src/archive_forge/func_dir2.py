import inspect
import types
def dir2(obj):
    """dir2(obj) -> list of strings

    Extended version of the Python builtin dir(), which does a few extra
    checks.

    This version is guaranteed to return only a list of true strings, whereas
    dir() returns anything that objects inject into themselves, even if they
    are later not really valid for attribute access (many extension libraries
    have such bugs).
    """
    try:
        words = set(dir(obj))
    except Exception:
        words = set()
    if safe_hasattr(obj, '__class__'):
        words |= set(dir(obj.__class__))
    words = [w for w in words if isinstance(w, str)]
    return sorted(words)