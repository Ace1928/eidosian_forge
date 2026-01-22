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
def correctly_signed_message(self, decoded_xml, msgtype, must=False, origdoc=None, only_valid_cert=False):
    """Check if a request is correctly signed, if we have metadata for
        the entity that sent the info use that, if not use the key that are in
        the message if any.

        :param decoded_xml: The SAML message as an XML infoset (a string)
        :param msgtype: SAML protocol message type
        :param must: Whether there must be a signature
        :param origdoc:
        :return:
        """
    attr = f'{msgtype}_from_string'
    _func = getattr(saml, attr, None)
    _func = getattr(samlp, attr, _func)
    msg = _func(decoded_xml)
    if not msg:
        raise TypeError(f'Not a {msgtype}')
    if not msg.signature:
        if must:
            err_msg = 'Required signature missing on {type}'
            err_msg = err_msg.format(type=msgtype)
            raise SignatureError(err_msg)
        else:
            return msg
    return self._check_signature(decoded_xml, msg, class_name(msg), origdoc, must=must, only_valid_cert=only_valid_cert)