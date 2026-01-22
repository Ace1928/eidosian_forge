def hack_select():
    """The python2.6 version of SocketServer does not handle interrupt calls
    from signals. Patch the select call if necessary.
    """
    import sys
    if sys.version_info[0] == 2 and sys.version_info[1] == 6:
        import select
        import errno
        real_select = select.select

        def _eintr_retry(*args):
            """restart a system call interrupted by EINTR"""
            while True:
                try:
                    return real_select(*args)
                except (OSError, select.error) as ex:
                    if ex.args[0] != errno.EINTR:
                        raise
        select.select = _eintr_retry