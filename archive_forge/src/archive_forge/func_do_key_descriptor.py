from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import class_name
from saml2 import md
from saml2 import samlp
from saml2 import xmldsig as ds
from saml2.algsupport import algorithm_support_in_metadata
from saml2.attribute_converter import from_local_name
from saml2.cert import read_cert_from_file
from saml2.config import Config
from saml2.extension import idpdisc
from saml2.extension import mdattr
from saml2.extension import mdui
from saml2.extension import shibmd
from saml2.extension import sp_type
from saml2.md import AttributeProfile
from saml2.s_utils import factory
from saml2.s_utils import rec_factory
from saml2.s_utils import sid
from saml2.saml import NAME_FORMAT_URI
from saml2.saml import Attribute
from saml2.saml import AttributeValue
from saml2.sigver import pre_signature_part
from saml2.sigver import security_context
from saml2.time_util import in_a_while
from saml2.validate import valid_instance
def do_key_descriptor(cert=None, enc_cert=None, use='both'):
    kd_list = []
    if use in ['signing', 'both'] and cert is not None:
        if not isinstance(cert, list):
            cert = [cert]
        for _cert in cert:
            kd_list.append(md.KeyDescriptor(key_info=ds.KeyInfo(x509_data=ds.X509Data(x509_certificate=ds.X509Certificate(text=_cert))), use='signing'))
    if use in ['both', 'encryption'] and enc_cert is not None:
        if not isinstance(enc_cert, list):
            enc_cert = [enc_cert]
        for _enc_cert in enc_cert:
            kd_list.append(md.KeyDescriptor(key_info=ds.KeyInfo(x509_data=ds.X509Data(x509_certificate=ds.X509Certificate(text=_enc_cert))), use='encryption'))
    if len(kd_list) == 0 and cert is not None:
        return md.KeyDescriptor(key_info=ds.KeyInfo(x509_data=ds.X509Data(x509_certificate=ds.X509Certificate(text=cert))))
    return kd_list