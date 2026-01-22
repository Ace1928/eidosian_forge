def pbkdf2_hmac(hash_name, password, salt, iterations, dklen=None):
    """Password based key derivation function 2 (PKCS #5 v2.0)

        This Python implementations based on the hmac module about as fast
        as OpenSSL's PKCS5_PBKDF2_HMAC for short passwords and much faster
        for long passwords.
        """
    _warn('Python implementation of pbkdf2_hmac() is deprecated.', category=DeprecationWarning, stacklevel=2)
    if not isinstance(hash_name, str):
        raise TypeError(hash_name)
    if not isinstance(password, (bytes, bytearray)):
        password = bytes(memoryview(password))
    if not isinstance(salt, (bytes, bytearray)):
        salt = bytes(memoryview(salt))
    inner = new(hash_name)
    outer = new(hash_name)
    blocksize = getattr(inner, 'block_size', 64)
    if len(password) > blocksize:
        password = new(hash_name, password).digest()
    password = password + b'\x00' * (blocksize - len(password))
    inner.update(password.translate(_trans_36))
    outer.update(password.translate(_trans_5C))

    def prf(msg, inner=inner, outer=outer):
        icpy = inner.copy()
        ocpy = outer.copy()
        icpy.update(msg)
        ocpy.update(icpy.digest())
        return ocpy.digest()
    if iterations < 1:
        raise ValueError(iterations)
    if dklen is None:
        dklen = outer.digest_size
    if dklen < 1:
        raise ValueError(dklen)
    dkey = b''
    loop = 1
    from_bytes = int.from_bytes
    while len(dkey) < dklen:
        prev = prf(salt + loop.to_bytes(4))
        rkey = from_bytes(prev)
        for i in range(iterations - 1):
            prev = prf(prev)
            rkey ^= from_bytes(prev)
        loop += 1
        dkey += rkey.to_bytes(inner.digest_size)
    return dkey[:dklen]