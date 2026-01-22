from __future__ import absolute_import
import math
import struct
import dns.inet
def option_from_wire(otype, wire, current, olen):
    """Build an EDNS option object from wire format.

    *otype*, an ``int``, is the option type.

    *wire*, a ``binary``, is the wire-format message.

    *current*, an ``int``, is the offset in *wire* of the beginning
    of the rdata.

    *olen*, an ``int``, is the length of the wire-format option data

    Returns an instance of a subclass of ``dns.edns.Option``.
    """
    cls = get_option_class(otype)
    return cls.from_wire(otype, wire, current, olen)