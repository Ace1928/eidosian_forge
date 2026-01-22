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
def set_opcode(self, opcode):
    """Set the opcode.

        *opcode*, an ``int``, is the opcode to set.
        """
    self.flags &= 34815
    self.flags |= dns.opcode.to_flags(opcode)