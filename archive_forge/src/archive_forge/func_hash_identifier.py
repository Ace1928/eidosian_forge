import operator
def hash_identifier(s, length, pad=True, hasher=md5, prefix='', group=None, upper=False):
    """
    Hashes the string (with the given hashing module), then turns that
    hash into an identifier of the given length (using modulo to
    reduce the length of the identifier).  If ``pad`` is False, then
    the minimum-length identifier will be used; otherwise the
    identifier will be padded with 0's as necessary.

    ``prefix`` will be added last, and does not count towards the
    target length.  ``group`` will group the characters with ``-`` in
    the given lengths, and also does not count towards the target
    length.  E.g., ``group=4`` will cause a identifier like
    ``a5f3-hgk3-asdf``.  Grouping occurs before the prefix.
    """
    if not callable(hasher):
        hasher = hasher.new
    if length > 26 and hasher is md5:
        raise ValueError('md5 cannot create hashes longer than 26 characters in length (you gave %s)' % length)
    if isinstance(s, str):
        s = s.encode('utf-8')
    elif not isinstance(s, bytes):
        s = str(s)
        s = s.encode('utf-8')
    h = hasher(s)
    bin_hash = h.digest()
    modulo = base ** length
    number = 0
    for c in list(bin_hash):
        number = (number * 256 + byte2int([c])) % modulo
    ident = make_identifier(number)
    if pad:
        ident = good_characters[0] * (length - len(ident)) + ident
    if group:
        parts = []
        while ident:
            parts.insert(0, ident[-group:])
            ident = ident[:-group]
        ident = '-'.join(parts)
    if upper:
        ident = ident.upper()
    return prefix + ident