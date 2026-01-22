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
def do_requested_attribute(attributes, acs, is_required='false', name_format=NAME_FORMAT_URI):
    lista = []
    for attr in attributes:
        attr = from_local_name(acs, attr, name_format)
        args = {}
        if isinstance(attr, str):
            args['name'] = attr
        else:
            for key in attr.keyswv():
                args[key] = getattr(attr, key)
        args['is_required'] = is_required
        args['name_format'] = name_format
        lista.append(md.RequestedAttribute(**args))
    return lista