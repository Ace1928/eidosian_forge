import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
def waitpid(self, pid, options):
    """
        Override C{os.waitpid}. Return values meaning that the child process
        has exited, save executed action.
        """
    self.actions.append('waitpid')
    if self.raiseWaitPid is not None:
        raise self.raiseWaitPid
    if self.waitChild is not None:
        return self.waitChild
    return (1, 0)