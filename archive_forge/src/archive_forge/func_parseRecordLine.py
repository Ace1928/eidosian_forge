import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
def parseRecordLine(self, origin, ttl, line):
    """
        Parse a C{line} from a zone file respecting C{origin} and C{ttl}.

        Add resulting records to authority.

        @param origin: starting point for the zone
        @type origin: L{bytes}

        @param ttl: time to live for the record
        @type ttl: L{int}

        @param line: zone file line to parse; split by word
        @type line: L{list} of L{bytes}
        """
    queryClasses = {qc.encode('ascii') for qc in dns.QUERY_CLASSES.values()}
    queryTypes = {qt.encode('ascii') for qt in dns.QUERY_TYPES.values()}
    markers = queryClasses | queryTypes
    cls = b'IN'
    owner = origin
    if line[0] == b'@':
        line = line[1:]
        owner = origin
    elif not line[0].isdigit() and line[0] not in markers:
        owner = line[0]
        line = line[1:]
    if line[0].isdigit() or line[0] in markers:
        domain = owner
        owner = origin
    else:
        domain = line[0]
        line = line[1:]
    if line[0] in queryClasses:
        cls = line[0]
        line = line[1:]
        if line[0].isdigit():
            ttl = int(line[0])
            line = line[1:]
    elif line[0].isdigit():
        ttl = int(line[0])
        line = line[1:]
        if line[0] in queryClasses:
            cls = line[0]
            line = line[1:]
    type = line[0]
    rdata = line[1:]
    self.addRecord(owner, ttl, nativeString(type), domain, nativeString(cls), rdata)