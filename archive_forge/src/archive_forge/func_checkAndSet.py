from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def checkAndSet(self, key, val, cas, flags=0, expireTime=0):
    """
        Change the content of C{key} only if the C{cas} value matches the
        current one associated with the key. Use this to store a value which
        hasn't been modified since last time you fetched it.

        @param key: The key to set.
        @type key: L{bytes}

        @param val: The value associated with the key.
        @type val: L{bytes}

        @param cas: Unique 64-bit value returned by previous call of C{get}.
        @type cas: L{bytes}

        @param flags: The flags to store with the key.
        @type flags: L{int}

        @param expireTime: If different from 0, the relative time in seconds
            when the key will be deleted from the store.
        @type expireTime: L{int}

        @return: A deferred that will fire with C{True} if the operation has
            succeeded, C{False} otherwise.
        @rtype: L{Deferred}
        """
    return self._set(b'cas', key, val, flags, expireTime, cas)