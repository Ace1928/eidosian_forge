import os
import platform
def ntob(n, encoding='ISO-8859-1'):
    """Return the native string as bytes in the given encoding."""
    assert_native(n)
    return n.encode(encoding)