import re
import dns.exception
def register_type(rdtype, rdtype_text, is_singleton=False):
    """Dynamically register an rdatatype.

    *rdtype*, an ``int``, the rdatatype to register.

    *rdtype_text*, a ``text``, the textual form of the rdatatype.

    *is_singleton*, a ``bool``, indicating if the type is a singleton (i.e.
    RRsets of the type can have only one member.)
    """
    _by_text[rdtype_text] = rdtype
    _by_value[rdtype] = rdtype_text
    if is_singleton:
        _singletons[rdtype] = True