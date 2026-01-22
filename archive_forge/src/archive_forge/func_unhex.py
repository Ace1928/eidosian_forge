def unhex(s):
    """Get the integer value of a hexadecimal number."""
    bits = 0
    for c in s:
        c = bytes((c,))
        if b'0' <= c <= b'9':
            i = ord('0')
        elif b'a' <= c <= b'f':
            i = ord('a') - 10
        elif b'A' <= c <= b'F':
            i = ord(b'A') - 10
        else:
            assert False, 'non-hex digit ' + repr(c)
        bits = bits * 16 + (ord(c) - i)
    return bits