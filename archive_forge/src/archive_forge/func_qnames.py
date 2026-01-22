import socket
import sys
import time
import random
import dns.exception
import dns.flags
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig
from ._compat import xrange, string_types
def qnames(self):
    """All of the names that were tried.

        Returns a list of ``dns.name.Name``.
        """
    return self.kwargs['qnames']