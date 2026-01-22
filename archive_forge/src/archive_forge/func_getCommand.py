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
def getCommand(self, commandName):
    """
        Return the path of the shell command named C{commandName}, looking at
        common locations.
        """
    for loc in procutils.which(commandName):
        return FilePath(loc).asBytesMode().path
    binLoc = FilePath('/bin').child(commandName)
    usrbinLoc = FilePath('/usr/bin').child(commandName)
    if binLoc.exists():
        return binLoc.asBytesMode().path
    elif usrbinLoc.exists():
        return usrbinLoc.asBytesMode().path
    else:
        raise RuntimeError(f'{commandName} found in neither standard location nor on PATH ({os.environ['PATH']})')