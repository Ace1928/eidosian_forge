from pyglet import compat_platform
def symbol_string(symbol):
    """Return a string describing a key symbol.

    Example::

        >>> symbol_string(BACKSPACE)
        'BACKSPACE'

    :Parameters:
        `symbol` : int
            Symbolic key constant.

    :rtype: str
    """
    if symbol < 1 << 32:
        return _key_names.get(symbol, str(symbol))
    else:
        return 'user_key(%x)' % (symbol >> 32)