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
def delete_rdataset(self, name, rdtype, covers=dns.rdatatype.NONE):
    """Delete the rdataset matching I{rdtype} and I{covers}, if it
        exists at the node specified by I{name}.

        The I{name}, I{rdtype}, and I{covers} parameters may be
        strings, in which case they will be converted to their proper
        type.

        It is not an error if the node does not exist, or if there is no
        matching rdataset at the node.

        If the node has no rdatasets after the deletion, it will itself
        be deleted.

        @param name: the owner name to look for
        @type name: DNS.name.Name object or string
        @param rdtype: the rdata type desired
        @type rdtype: int or string
        @param covers: the covered type (defaults to None)
        @type covers: int or string
        """
    name = self._validate_name(name)
    if isinstance(rdtype, string_types):
        rdtype = dns.rdatatype.from_text(rdtype)
    if isinstance(covers, string_types):
        covers = dns.rdatatype.from_text(covers)
    node = self.get_node(name)
    if node is not None:
        node.delete_rdataset(self.rdclass, rdtype, covers)
        if len(node) == 0:
            self.delete_node(name)