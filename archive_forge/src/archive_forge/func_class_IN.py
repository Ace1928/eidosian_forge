import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
def class_IN(self, ttl, type, domain, rdata):
    """
        Simulate a class IN and recurse into the actual class.

        @param ttl: time to live for the record
        @type ttl: L{int}

        @param type: record type
        @type type: str

        @param domain: the domain
        @type domain: bytes

        @param rdata:
        @type rdata: bytes
        """
    record = getattr(dns, f'Record_{nativeString(type)}', None)
    if record:
        r = record(*rdata)
        r.ttl = ttl
        self.records.setdefault(domain.lower(), []).append(r)
        if type == 'SOA':
            self.soa = (domain, r)
    else:
        raise NotImplementedError(f'Record type {nativeString(type)!r} not supported')