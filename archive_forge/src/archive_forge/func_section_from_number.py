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
def section_from_number(self, number):
    """Return the "section number" of the specified section for use
        in indexing.  The question section is 0, the answer section is 1,
        the authority section is 2, and the additional section is 3.

        *section* is one of the section attributes of this message.

        Raises ``ValueError`` if the section isn't known.

        Returns an ``int``.
        """
    if number == QUESTION:
        return self.question
    elif number == ANSWER:
        return self.answer
    elif number == AUTHORITY:
        return self.authority
    elif number == ADDITIONAL:
        return self.additional
    else:
        raise ValueError('unknown section')