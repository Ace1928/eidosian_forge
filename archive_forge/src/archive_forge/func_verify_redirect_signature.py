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
def verify_redirect_signature(saml_msg, crypto, cert=None, sigkey=None):
    """

    :param saml_msg: A dictionary with strings as values, *NOT* lists as
    produced by parse_qs.
    :param cert: A certificate to use when verifying the signature
    :return: True, if signature verified
    """
    try:
        signer = crypto.get_signer(saml_msg['SigAlg'], sigkey)
    except KeyError:
        raise Unsupported(f'Signature algorithm: {saml_msg['SigAlg']}')
    else:
        if saml_msg['SigAlg'] in SIGNER_ALGS:
            if 'SAMLRequest' in saml_msg:
                _order = REQ_ORDER
            elif 'SAMLResponse' in saml_msg:
                _order = RESP_ORDER
            else:
                raise Unsupported('Verifying signature on something that should not be signed')
            _args = saml_msg.copy()
            del _args['Signature']
            string = '&'.join([parse.urlencode({k: _args[k]}) for k in _order if k in _args]).encode('ascii')
            if cert:
                _key = extract_rsa_key_from_x509_cert(pem_format(cert))
            else:
                _key = sigkey
            _sign = base64.b64decode(saml_msg['Signature'])
            return bool(signer.verify(string, _sign, _key))