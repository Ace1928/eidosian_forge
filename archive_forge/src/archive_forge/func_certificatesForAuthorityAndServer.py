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
def certificatesForAuthorityAndServer(serviceIdentity='example.com'):
    """
    Create a self-signed CA certificate and server certificate signed by the
    CA.

    @param serviceIdentity: The identity (hostname) of the server.
    @type serviceIdentity: L{unicode}

    @return: a 2-tuple of C{(certificate_authority_certificate,
        server_certificate)}
    @rtype: L{tuple} of (L{sslverify.Certificate},
        L{sslverify.PrivateCertificate})
    """
    commonNameForCA = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Testing Example CA')])
    commonNameForServer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Testing Example Server')])
    oneDay = datetime.timedelta(1, 0, 0)
    privateKeyForCA = rsa.generate_private_key(public_exponent=65537, key_size=4096, backend=default_backend())
    publicKeyForCA = privateKeyForCA.public_key()
    caCertificate = x509.CertificateBuilder().subject_name(commonNameForCA).issuer_name(commonNameForCA).not_valid_before(datetime.datetime.today() - oneDay).not_valid_after(datetime.datetime.today() + oneDay).serial_number(x509.random_serial_number()).public_key(publicKeyForCA).add_extension(x509.BasicConstraints(ca=True, path_length=9), critical=True).sign(private_key=privateKeyForCA, algorithm=hashes.SHA256(), backend=default_backend())
    privateKeyForServer = rsa.generate_private_key(public_exponent=65537, key_size=4096, backend=default_backend())
    publicKeyForServer = privateKeyForServer.public_key()
    try:
        ipAddress = ipaddress.ip_address(serviceIdentity)
    except ValueError:
        subjectAlternativeNames = [x509.DNSName(serviceIdentity.encode('idna').decode('ascii'))]
    else:
        subjectAlternativeNames = [x509.IPAddress(ipAddress)]
    serverCertificate = x509.CertificateBuilder().subject_name(commonNameForServer).issuer_name(commonNameForCA).not_valid_before(datetime.datetime.today() - oneDay).not_valid_after(datetime.datetime.today() + oneDay).serial_number(x509.random_serial_number()).public_key(publicKeyForServer).add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True).add_extension(x509.SubjectAlternativeName(subjectAlternativeNames), critical=True).sign(private_key=privateKeyForCA, algorithm=hashes.SHA256(), backend=default_backend())
    caSelfCert = sslverify.Certificate.loadPEM(caCertificate.public_bytes(Encoding.PEM))
    serverCert = sslverify.PrivateCertificate.loadPEM(b'\n'.join([privateKeyForServer.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()), serverCertificate.public_bytes(Encoding.PEM)]))
    return (caSelfCert, serverCert)