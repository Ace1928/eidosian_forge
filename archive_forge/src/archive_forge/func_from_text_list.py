import dns.name
import dns.rdataset
import dns.rdataclass
import dns.renderer
from ._compat import string_types
def from_text_list(name, ttl, rdclass, rdtype, text_rdatas, idna_codec=None):
    """Create an RRset with the specified name, TTL, class, and type, and with
    the specified list of rdatas in text format.

    Returns a ``dns.rrset.RRset`` object.
    """
    if isinstance(name, string_types):
        name = dns.name.from_text(name, None, idna_codec=idna_codec)
    if isinstance(rdclass, string_types):
        rdclass = dns.rdataclass.from_text(rdclass)
    if isinstance(rdtype, string_types):
        rdtype = dns.rdatatype.from_text(rdtype)
    r = RRset(name, rdclass, rdtype)
    r.update_ttl(ttl)
    for t in text_rdatas:
        rd = dns.rdata.from_text(r.rdclass, r.rdtype, t)
        r.add(rd)
    return r