import base64
import datetime
import ssl
from urllib.parse import urljoin, urlparse
import cryptography.hazmat.primitives.hashes
import requests
from cryptography import hazmat, x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric.dsa import DSAPublicKey
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA, EllipticCurvePublicKey
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.hashes import SHA1, Hash
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.x509 import ocsp
from redis.exceptions import AuthorizationError, ConnectionError
def components_from_direct_connection(self):
    """Return the certificate, primary issuer, and primary ocsp server
        from the host defined by the socket. This is useful in cases where
        different certificates are occasionally presented.
        """
    pem = ssl.get_server_certificate((self.HOST, self.PORT), ca_certs=self.CA_CERTS)
    cert = x509.load_pem_x509_certificate(pem.encode(), backends.default_backend())
    return self._certificate_components(cert)