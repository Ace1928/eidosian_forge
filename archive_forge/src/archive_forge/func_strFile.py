def strFile(p, f, caseSensitive=True):
    """
    Find whether string C{p} occurs in a read()able object C{f}.

    @rtype: C{bool}
    """
    buf = type(p)()
    buf_len = max(len(p), 2 ** 2 ** 2 ** 2)
    if not caseSensitive:
        p = p.lower()
    while 1:
        r = f.read(buf_len - len(p))
        if not caseSensitive:
            r = r.lower()
        bytes_read = len(r)
        if bytes_read == 0:
            return False
        l = len(buf) + bytes_read - buf_len
        if l <= 0:
            buf = buf + r
        else:
            buf = buf[l:] + r
        if buf.find(p) != -1:
            return True