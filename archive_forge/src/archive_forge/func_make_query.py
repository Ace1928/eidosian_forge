from __future__ import absolute_import
from io import StringIO
import struct
import time
import dns.edns
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.entropy
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rrset
import dns.renderer
import dns.tsig
import dns.wiredata
from ._compat import long, xrange, string_types
def make_query(qname, rdtype, rdclass=dns.rdataclass.IN, use_edns=None, want_dnssec=False, ednsflags=None, payload=None, request_payload=None, options=None):
    """Make a query message.

    The query name, type, and class may all be specified either
    as objects of the appropriate type, or as strings.

    The query will have a randomly chosen query id, and its DNS flags
    will be set to dns.flags.RD.

    qname, a ``dns.name.Name`` or ``text``, the query name.

    *rdtype*, an ``int`` or ``text``, the desired rdata type.

    *rdclass*, an ``int`` or ``text``,  the desired rdata class; the default
    is class IN.

    *use_edns*, an ``int``, ``bool`` or ``None``.  The EDNS level to use; the
    default is None (no EDNS).
    See the description of dns.message.Message.use_edns() for the possible
    values for use_edns and their meanings.

    *want_dnssec*, a ``bool``.  If ``True``, DNSSEC data is desired.

    *ednsflags*, an ``int``, the EDNS flag values.

    *payload*, an ``int``, is the EDNS sender's payload field, which is the
    maximum size of UDP datagram the sender can handle.  I.e. how big
    a response to this message can be.

    *request_payload*, an ``int``, is the EDNS payload size to use when
    sending this message.  If not specified, defaults to the value of
    *payload*.

    *options*, a list of ``dns.edns.Option`` objects or ``None``, the EDNS
    options.

    Returns a ``dns.message.Message``
    """
    if isinstance(qname, string_types):
        qname = dns.name.from_text(qname)
    if isinstance(rdtype, string_types):
        rdtype = dns.rdatatype.from_text(rdtype)
    if isinstance(rdclass, string_types):
        rdclass = dns.rdataclass.from_text(rdclass)
    m = Message()
    m.flags |= dns.flags.RD
    m.find_rrset(m.question, qname, rdclass, rdtype, create=True, force_unique=True)
    kwargs = {}
    if ednsflags is not None:
        kwargs['ednsflags'] = ednsflags
        if use_edns is None:
            use_edns = 0
    if payload is not None:
        kwargs['payload'] = payload
        if use_edns is None:
            use_edns = 0
    if request_payload is not None:
        kwargs['request_payload'] = request_payload
        if use_edns is None:
            use_edns = 0
    if options is not None:
        kwargs['options'] = options
        if use_edns is None:
            use_edns = 0
    kwargs['edns'] = use_edns
    m.use_edns(**kwargs)
    m.want_dnssec(want_dnssec)
    return m