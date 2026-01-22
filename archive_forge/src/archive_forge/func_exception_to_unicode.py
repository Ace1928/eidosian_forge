import sys
def exception_to_unicode(exc):
    """Get the message of an exception as a Unicode string.

    On Python 3, the exception message is always a Unicode string. On
    Python 2, the exception message is a bytes string *most* of the time.

    If the exception message is a bytes strings, try to decode it from UTF-8
    (superset of ASCII), from the locale encoding, or fallback to decoding it
    from ISO-8859-1 (which never fails).

    .. versionadded:: 1.6
    """
    msg = None
    if msg is None:
        msg = exc.__str__()
    if isinstance(msg, str):
        return msg
    try:
        return msg.decode('utf-8')
    except UnicodeDecodeError:
        pass
    encoding = _getfilesystemencoding()
    try:
        return msg.decode(encoding)
    except UnicodeDecodeError:
        pass
    return msg.decode('latin1')