import re
import dns.exception
Dynamically register an rdatatype.

    *rdtype*, an ``int``, the rdatatype to register.

    *rdtype_text*, a ``text``, the textual form of the rdatatype.

    *is_singleton*, a ``bool``, indicating if the type is a singleton (i.e.
    RRsets of the type can have only one member.)
    