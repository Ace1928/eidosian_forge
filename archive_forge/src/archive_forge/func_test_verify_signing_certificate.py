import datetime
import mock
import os
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cursive import certificate_utils
from cursive import exception
from cursive.tests import base
def test_verify_signing_certificate(self):
    signing_certificate = self.load_certificate('self_signed_cert.pem')
    signed_certificate = self.load_certificate('signed_cert.pem')
    certificate_utils.verify_certificate_signature(signing_certificate, signed_certificate)