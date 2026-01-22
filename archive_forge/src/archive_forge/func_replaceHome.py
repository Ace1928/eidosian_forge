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
def replaceHome(self, tempHome):
    """
        Replace the HOME environment variable until the end of the current
        test, with the given new home-directory, so that L{os.path.expanduser}
        will yield controllable, predictable results.

        @param tempHome: the pathname to replace the HOME variable with.

        @type tempHome: L{str}
        """
    oldHome = os.environ.get('HOME')

    def cleanupHome():
        if oldHome is None:
            del os.environ['HOME']
        else:
            os.environ['HOME'] = oldHome
    self.addCleanup(cleanupHome)
    os.environ['HOME'] = tempHome