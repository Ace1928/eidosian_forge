import operator

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
    