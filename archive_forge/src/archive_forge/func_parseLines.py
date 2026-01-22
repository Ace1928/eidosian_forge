import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
def parseLines(self, lines):
    """
        Parse C{lines}.

        @param lines: lines to work on
        @type lines: iterable of L{bytes}
        """
    ttl = 60 * 60 * 3
    origin = self.origin
    self.records = {}
    for line in lines:
        if line[0] == b'$TTL':
            ttl = dns.str2time(line[1])
        elif line[0] == b'$ORIGIN':
            origin = line[1]
        elif line[0] == b'$INCLUDE':
            raise NotImplementedError('$INCLUDE directive not implemented')
        elif line[0] == b'$GENERATE':
            raise NotImplementedError('$GENERATE directive not implemented')
        else:
            self.parseRecordLine(origin, ttl, line)
    self.origin = origin