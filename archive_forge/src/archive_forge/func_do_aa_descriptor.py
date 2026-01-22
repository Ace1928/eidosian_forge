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
def do_aa_descriptor(conf, cert=None, enc_cert=None):
    aad = md.AttributeAuthorityDescriptor()
    aad.protocol_support_enumeration = samlp.NAMESPACE
    exts = conf.getattr('extensions', 'aa')
    if exts:
        if aad.extensions is None:
            aad.extensions = md.Extensions()
        for key, val in exts.items():
            _ext = do_extensions(key, val)
            if _ext:
                for _e in _ext:
                    aad.extensions.add_extension_element(_e)
    endps = conf.getattr('endpoints', 'aa')
    if endps:
        for endpoint, instlist in do_endpoints(endps, ENDPOINTS['aa']).items():
            setattr(aad, endpoint, instlist)
    _do_nameid_format(aad, conf, 'aa')
    if cert or enc_cert:
        aad.key_descriptor = do_key_descriptor(cert, enc_cert, use=conf.metadata_key_usage)
    attributes = conf.getattr('attribute', 'aa')
    if attributes:
        for attribute in attributes:
            aad.attribute.append(Attribute(text=attribute))
    attribute_profiles = conf.getattr('attribute_profile', 'aa')
    if attribute_profiles:
        for attribute_profile in attribute_profiles:
            aad.attribute.append(AttributeProfile(text=attribute_profile))
    return aad