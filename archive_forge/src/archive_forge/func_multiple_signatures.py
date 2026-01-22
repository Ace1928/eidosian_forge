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
def multiple_signatures(self, statement, to_sign, key=None, key_file=None, sign_alg=None, digest_alg=None):
    """
        Sign multiple parts of a statement

        :param statement: The statement that should be sign, this is XML text
        :param to_sign: A list of (items, id) tuples that specifies what to sign
        :param key: A key that should be used for doing the signing
        :param key_file: A file that contains the key to be used
        :return: A possibly multiple signed statement
        """
    for item, sid in to_sign:
        if not sid:
            if not item.id:
                sid = item.id = sid()
            else:
                sid = item.id
        if not item.signature:
            item.signature = pre_signature_part(ident=sid, public_key=self.cert_file, sign_alg=sign_alg, digest_alg=digest_alg)
        statement = self.sign_statement(statement, class_name(item), key=key, key_file=key_file, node_id=sid)
    return statement