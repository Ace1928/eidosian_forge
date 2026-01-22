import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
def generateCertificateFiles(basename, organization, organizationalUnit):
    """
    Create certificate files key, req and cert prefixed by C{basename} for
    given C{organization} and C{organizationalUnit}.
    """
    pkey, req, cert = generateCertificateObjects(organization, organizationalUnit)
    for ext, obj, dumpFunc in [('key', pkey, crypto.dump_privatekey), ('req', req, crypto.dump_certificate_request), ('cert', cert, crypto.dump_certificate)]:
        fName = os.extsep.join((basename, ext)).encode('utf-8')
        FilePath(fName).setContent(dumpFunc(crypto.FILETYPE_PEM, obj))