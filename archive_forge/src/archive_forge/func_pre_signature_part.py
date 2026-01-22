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
def pre_signature_part(ident, public_key=None, identifier=None, digest_alg=None, sign_alg=None):
    """
    If an assertion is to be signed the signature part has to be preset
    with which algorithms to be used, this function returns such a
    preset part.

    :param ident: The identifier of the assertion, so you know which assertion
        was signed
    :param public_key: The base64 part of a PEM file
    :param identifier:
    :return: A preset signature part
    """
    if not digest_alg:
        digest_alg = ds.DefaultSignature().get_digest_alg()
    if not sign_alg:
        sign_alg = ds.DefaultSignature().get_sign_alg()
    signature_method = ds.SignatureMethod(algorithm=sign_alg)
    canonicalization_method = ds.CanonicalizationMethod(algorithm=TRANSFORM_C14N)
    trans0 = ds.Transform(algorithm=TRANSFORM_ENVELOPED)
    trans1 = ds.Transform(algorithm=TRANSFORM_C14N)
    transforms = ds.Transforms(transform=[trans0, trans1])
    digest_method = ds.DigestMethod(algorithm=digest_alg)
    reference = ds.Reference(uri=f'#{ident}', digest_value=ds.DigestValue(), transforms=transforms, digest_method=digest_method)
    signed_info = ds.SignedInfo(signature_method=signature_method, canonicalization_method=canonicalization_method, reference=reference)
    signature = ds.Signature(signed_info=signed_info, signature_value=ds.SignatureValue())
    if identifier:
        signature.id = f'Signature{identifier}'
    if public_key:
        x509_data = ds.X509Data(x509_certificate=[ds.X509Certificate(text=public_key)])
        key_info = ds.KeyInfo(x509_data=x509_data)
        signature.key_info = key_info
    return signature