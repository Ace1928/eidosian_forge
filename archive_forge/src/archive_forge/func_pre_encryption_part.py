import base64
import datetime
import hashlib
import itertools
import logging
import os
import re
from subprocess import PIPE
from subprocess import Popen
import sys
from tempfile import NamedTemporaryFile
from time import mktime
from uuid import uuid4 as gen_random_key
import dateutil
from urllib import parse
from OpenSSL import crypto
import pytz
from saml2 import ExtensionElement
from saml2 import SamlBase
from saml2 import SAMLError
from saml2 import class_name
from saml2 import extension_elements_to_elements
from saml2 import saml
from saml2 import samlp
from saml2.cert import CertificateError
from saml2.cert import OpenSSLWrapper
from saml2.cert import read_cert_from_file
import saml2.cryptography.asymmetric
import saml2.cryptography.pki
import saml2.data.templates as _data_template
from saml2.extension import pefim
from saml2.extension.pefim import SPCertEnc
from saml2.s_utils import Unsupported
from saml2.saml import EncryptedAssertion
from saml2.time_util import str_to_time
from saml2.xml.schema import XMLSchemaError
from saml2.xml.schema import validate as validate_doc_with_schema
from saml2.xmldsig import ALLOWED_CANONICALIZATIONS
from saml2.xmldsig import ALLOWED_TRANSFORMS
from saml2.xmldsig import SIG_RSA_SHA1
from saml2.xmldsig import SIG_RSA_SHA224
from saml2.xmldsig import SIG_RSA_SHA256
from saml2.xmldsig import SIG_RSA_SHA384
from saml2.xmldsig import SIG_RSA_SHA512
from saml2.xmldsig import TRANSFORM_C14N
from saml2.xmldsig import TRANSFORM_ENVELOPED
import saml2.xmldsig as ds
from saml2.xmlenc import CipherData
from saml2.xmlenc import CipherValue
from saml2.xmlenc import EncryptedData
from saml2.xmlenc import EncryptedKey
from saml2.xmlenc import EncryptionMethod
def pre_encryption_part(*, msg_enc=TRIPLE_DES_CBC, key_enc=RSA_OAEP_MGF1P, key_name=None, encrypted_key_id=None, encrypted_data_id=None, encrypt_cert=None):
    ek_id = encrypted_key_id or f'EK_{gen_random_key()}'
    ed_id = encrypted_data_id or f'ED_{gen_random_key()}'
    msg_encryption_method = EncryptionMethod(algorithm=msg_enc)
    key_encryption_method = EncryptionMethod(algorithm=key_enc)
    x509_data = ds.X509Data(x509_certificate=ds.X509Certificate(text=encrypt_cert)) if encrypt_cert else None
    key_name = ds.KeyName(text=key_name) if key_name else None
    key_info = ds.KeyInfo(key_name=key_name, x509_data=x509_data) if key_name or x509_data else None
    encrypted_key = EncryptedKey(id=ek_id, encryption_method=key_encryption_method, key_info=key_info, cipher_data=CipherData(cipher_value=CipherValue(text='')))
    key_info = ds.KeyInfo(encrypted_key=encrypted_key)
    encrypted_data = EncryptedData(id=ed_id, type='http://www.w3.org/2001/04/xmlenc#Element', encryption_method=msg_encryption_method, key_info=key_info, cipher_data=CipherData(cipher_value=CipherValue(text='')))
    return encrypted_data