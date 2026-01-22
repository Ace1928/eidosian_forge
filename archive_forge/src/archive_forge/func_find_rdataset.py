from __future__ import generators
import sys
import re
import os
from io import BytesIO
import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdatatype
import dns.rdata
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.ttl
import dns.grange
from ._compat import string_types, text_type, PY3
def find_rdataset(self, name, rdtype, covers=dns.rdatatype.NONE, create=False):
    """Look for rdata with the specified name and type in the zone,
        and return an rdataset encapsulating it.

        The I{name}, I{rdtype}, and I{covers} parameters may be
        strings, in which case they will be converted to their proper
        type.

        The rdataset returned is not a copy; changes to it will change
        the zone.

        KeyError is raised if the name or type are not found.
        Use L{get_rdataset} if you want to have None returned instead.

        @param name: the owner name to look for
        @type name: DNS.name.Name object or string
        @param rdtype: the rdata type desired
        @type rdtype: int or string
        @param covers: the covered type (defaults to None)
        @type covers: int or string
        @param create: should the node and rdataset be created if they do not
        exist?
        @type create: bool
        @raises KeyError: the node or rdata could not be found
        @rtype: dns.rdataset.Rdataset object
        """
    name = self._validate_name(name)
    if isinstance(rdtype, string_types):
        rdtype = dns.rdatatype.from_text(rdtype)
    if isinstance(covers, string_types):
        covers = dns.rdatatype.from_text(covers)
    node = self.find_node(name, create)
    return node.find_rdataset(self.rdclass, rdtype, covers, create)