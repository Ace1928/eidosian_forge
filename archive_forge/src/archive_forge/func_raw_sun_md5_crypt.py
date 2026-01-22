from hashlib import md5
import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import to_unicode
from passlib.utils.binary import h64
from passlib.utils.compat import byte_elem_value, irange, u, \
import passlib.utils.handlers as uh
def raw_sun_md5_crypt(secret, rounds, salt):
    """given secret & salt, return encoded sun-md5-crypt checksum"""
    global MAGIC_HAMLET
    assert isinstance(secret, bytes)
    assert isinstance(salt, bytes)
    if rounds <= 0:
        rounds = 0
    real_rounds = 4096 + rounds
    result = md5(secret + salt).digest()
    assert len(result) == 16
    X_ROUNDS_0, X_ROUNDS_1, Y_ROUNDS_0, Y_ROUNDS_1 = _XY_ROUNDS
    round = 0
    while round < real_rounds:
        rval = [byte_elem_value(c) for c in result].__getitem__
        x = 0
        xrounds = X_ROUNDS_1 if rval(round >> 3 & 15) >> (round & 7) & 1 else X_ROUNDS_0
        for i, ia, ib in xrounds:
            a = rval(ia)
            b = rval(ib)
            v = rval(a >> b % 5 & 15) >> (b >> (a & 7) & 1)
            x |= (rval(v >> 3 & 15) >> (v & 7) & 1) << i
        y = 0
        yrounds = Y_ROUNDS_1 if rval(round + 64 >> 3 & 15) >> (round & 7) & 1 else Y_ROUNDS_0
        for i, ia, ib in yrounds:
            a = rval(ia)
            b = rval(ib)
            v = rval(a >> b % 5 & 15) >> (b >> (a & 7) & 1)
            y |= (rval(v >> 3 & 15) >> (v & 7) & 1) << i
        coin = (rval(x >> 3) >> (x & 7) ^ rval(y >> 3) >> (y & 7)) & 1
        h = md5(result)
        if coin:
            h.update(MAGIC_HAMLET)
        h.update(unicode(round).encode('ascii'))
        result = h.digest()
        round += 1
    return h64.encode_transposed_bytes(result, _chk_offsets)