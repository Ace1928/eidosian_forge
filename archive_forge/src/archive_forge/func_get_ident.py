def get_ident():
    """Dummy implementation of _thread.get_ident().

    Since this module should only be used when _threadmodule is not
    available, it is safe to assume that the current process is the
    only thread.  Thus a constant can be safely returned.
    """
    return -1