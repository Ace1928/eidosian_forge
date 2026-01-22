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
def create_metadata_string(configfile, config=None, valid=None, cert=None, keyfile=None, mid=None, name=None, sign=None, sign_alg=None, digest_alg=None):
    valid_for = 0
    nspair = {'xs': 'http://www.w3.org/2001/XMLSchema'}
    if valid:
        valid_for = int(valid)
    eds = []
    if config is None:
        if configfile.endswith('.py'):
            configfile = configfile[:-3]
        config = Config().load_file(configfile)
    eds.append(entity_descriptor(config))
    conf = Config()
    conf.key_file = config.key_file or keyfile
    conf.cert_file = config.cert_file or cert
    conf.xmlsec_binary = config.xmlsec_binary
    conf.crypto_backend = config.crypto_backend
    secc = security_context(conf)
    sign_alg = sign_alg or config.signing_algorithm
    digest_alg = digest_alg or config.digest_algorithm
    if mid:
        eid, xmldoc = entities_descriptor(eds, valid_for, name, mid, sign, secc, sign_alg, digest_alg)
    else:
        eid = eds[0]
        if sign:
            eid, xmldoc = sign_entity_descriptor(eid, mid, secc, sign_alg, digest_alg)
        else:
            xmldoc = None
    valid_instance(eid)
    return metadata_tostring_fix(eid, nspair, xmldoc)