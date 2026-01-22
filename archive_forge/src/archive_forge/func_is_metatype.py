import re
import dns.exception
def is_metatype(rdtype):
    """True if the specified type is a metatype.

    *rdtype* is an ``int``.

    The currently defined metatypes are TKEY, TSIG, IXFR, AXFR, MAILA,
    MAILB, ANY, and OPT.

    Returns a ``bool``.
    """
    if rdtype >= TKEY and rdtype <= ANY or rdtype in _metatypes:
        return True
    return False