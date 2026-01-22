import errno
import os
import socket
from unittest import skipIf
from twisted.internet import interfaces, reactor
from twisted.internet.defer import gatherResults, maybeDeferred
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.python import log
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@skipIf(platform.getType() == 'win32', 'Windows accept(2) cannot generate ENOMEM')
def test_noMemoryFromAccept(self):
    """
        Similar to L{test_tooManyFilesFromAccept}, but test the case where
        C{accept(2)} fails with C{ENOMEM}.

        On Linux at least, this can sensibly occur, even in a Python program
        (which eats memory like no ones business), when memory has become
        fragmented or low memory has been filled (d_alloc calls
        kmem_cache_alloc calls kmalloc - kmalloc only allocates out of low
        memory).
        """
    return self._acceptFailureTest(ENOMEM)