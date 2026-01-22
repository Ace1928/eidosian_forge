import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
def pathContainingDumpOf(testCase, *dumpables):
    """
    Create a temporary file to store some serializable-as-PEM objects in, and
    return its name.

    @param testCase: a test case to use for generating a temporary directory.
    @type testCase: L{twisted.trial.unittest.TestCase}

    @param dumpables: arguments are objects from pyOpenSSL with a C{dump}
        method, taking a pyOpenSSL file-type constant, such as
        L{OpenSSL.crypto.FILETYPE_PEM} or L{OpenSSL.crypto.FILETYPE_ASN1}.
    @type dumpables: L{tuple} of L{object} with C{dump} method taking L{int}
        returning L{bytes}

    @return: the path to a file where all of the dumpables were dumped in PEM
        format.
    @rtype: L{str}
    """
    fname = testCase.mktemp()
    with open(fname, 'wb') as f:
        for dumpable in dumpables:
            f.write(dumpable.dump(FILETYPE_PEM))
    return fname