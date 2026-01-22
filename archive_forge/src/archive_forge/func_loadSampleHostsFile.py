import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
def loadSampleHostsFile(self, content=sampleHashedLine + otherSamplePlaintextLine + b'\n# That was a blank line.\nThis is just unparseable.\n|1|This also unparseable.\n'):
    """
        Return a sample hosts file, with keys for www.twistedmatrix.com and
        divmod.com present.
        """
    return KnownHostsFile.fromPath(self.pathWithContent(content))