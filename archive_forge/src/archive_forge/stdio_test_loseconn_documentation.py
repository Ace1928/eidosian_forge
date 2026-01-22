import sys
from twisted.internet import protocol, stdio
from twisted.internet.error import ConnectionDone
from twisted.python import log, reflect

        Check that C{reason} is a L{Failure} wrapping a L{ConnectionDone}
        instance and stop the reactor.  If C{reason} is wrong for some reason,
        log something about that in C{self.errorLogFile} and make sure the
        process exits with a non-zero status.
        