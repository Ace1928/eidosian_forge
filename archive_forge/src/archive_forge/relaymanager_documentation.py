import email.utils
import os
import pickle
import time
from typing import Type
from twisted.application import internet
from twisted.internet import protocol
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import DNSLookupError
from twisted.internet.protocol import connectionDone
from twisted.mail import bounce, relay, smtp
from twisted.python import log
from twisted.python.failure import Failure

        Attempt to use the name of the domain directly when mail exchange
        lookup fails.

        @type failure: L{Failure}
        @param failure: The reason for the lookup failure.

        @type domain: L{bytes}
        @param domain: The domain name.

        @rtype: L{Record_MX <twisted.names.dns.Record_MX>} or L{Failure}
        @return: An MX record for the domain or a failure if the fallback to
            domain option is not in effect and an error, other than not
            finding an MX record, occurred during lookup.

        @raise IOError: When no MX record could be found and the fallback to
            domain option is not in effect.

        @raise DNSLookupError: When no MX record could be found and the
            fallback to domain option is in effect but no address for the
            domain could be found.
        