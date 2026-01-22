from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def getMultiple(self, keys, withIdentifier=False):
    """
        Get the given list of C{keys}.  If C{withIdentifier} is set to C{True},
        the command issued is a C{gets}, that will return the identifiers
        associated with each values. This identifier has to be used when
        issuing C{checkAndSet} update later, using the corresponding method.

        @param keys: The keys to retrieve.
        @type keys: L{list} of L{bytes}

        @param withIdentifier: If set to C{True}, retrieve the identifiers
            along with the values and the flags.
        @type withIdentifier: L{bool}

        @return: A deferred that will fire with a dictionary with the elements
            of C{keys} as keys and the tuples (flags, value) as values if
            C{withIdentifier} is C{False}, or (flags, cas identifier, value) if
            C{True}.  If the server indicates there is no value associated with
            C{key}, the returned values will be L{None} and the returned flags
            will be C{0}.
        @rtype: L{Deferred}

        @since: 9.0
        """
    return self._get(keys, withIdentifier, True)