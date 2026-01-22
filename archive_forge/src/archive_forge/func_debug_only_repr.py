def debug_only_repr(value, param='hash'):
    """
    helper used to display sensitive data (hashes etc) within error messages.
    currently returns placeholder test UNLESS unittests are running,
    in which case the real value is displayed.

    mainly useful to prevent hashes / secrets from being exposed in production tracebacks;
    while still being visible from test failures.

    NOTE: api subject to change, may formalize this more in the future.
    """
    if ENABLE_DEBUG_ONLY_REPR or value is None or isinstance(value, bool):
        return repr(value)
    return '<%s %s value omitted>' % (param, type(value))