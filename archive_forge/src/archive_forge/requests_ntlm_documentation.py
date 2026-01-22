import warnings
import base64
import typing as t
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import UnsupportedAlgorithm
from requests.auth import AuthBase
from requests.packages.urllib3.response import HTTPResponse
import spnego

        Get the certificate at the request_url and return it as a hash. Will get the raw socket from the
        original response from the server. This socket is then checked if it is an SSL socket and then used to
        get the hash of the certificate. The certificate hash is then used with NTLMv2 authentication for
        Channel Binding Tokens support. If the raw object is not a urllib3 HTTPReponse (default with requests)
        then no certificate will be returned.

        :param response: The original 401 response from the server
        :return: The hash of the DER encoded certificate at the request_url or None if not a HTTPS endpoint
        