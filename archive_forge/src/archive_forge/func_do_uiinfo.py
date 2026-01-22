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
def do_uiinfo(_uiinfo):
    uii = mdui.UIInfo()
    for attr in ['display_name', 'description', 'information_url', 'privacy_statement_url']:
        try:
            val = _uiinfo[attr]
        except KeyError:
            continue
        aclass = uii.child_class(attr)
        inst = getattr(uii, attr)
        if isinstance(val, str):
            ainst = aclass(text=val)
            inst.append(ainst)
        elif isinstance(val, dict):
            ainst = aclass()
            ainst.text = val['text']
            ainst.lang = val['lang']
            inst.append(ainst)
        else:
            for value in val:
                if isinstance(value, str):
                    ainst = aclass(text=value)
                    inst.append(ainst)
                elif isinstance(value, dict):
                    ainst = aclass()
                    ainst.text = value['text']
                    ainst.lang = value['lang']
                    inst.append(ainst)
    try:
        _attr = 'logo'
        val = _uiinfo[_attr]
        inst = getattr(uii, _attr)
        if isinstance(val, dict):
            logo = mdui.Logo()
            for attr, value in val.items():
                if attr in logo.keys():
                    setattr(logo, attr, value)
            inst.append(logo)
        elif isinstance(val, list):
            for logga in val:
                if not isinstance(logga, dict):
                    raise SAMLError('Configuration error !!')
                logo = mdui.Logo()
                for attr, value in logga.items():
                    if attr in logo.keys():
                        setattr(logo, attr, value)
                inst.append(logo)
    except KeyError:
        pass
    try:
        _attr = 'keywords'
        val = _uiinfo[_attr]
        inst = getattr(uii, _attr)
        if isinstance(val, list):
            for value in val:
                keyw = mdui.Keywords()
                if isinstance(value, str):
                    keyw.text = value
                elif isinstance(value, dict):
                    keyw.text = ' '.join(value['text'])
                    try:
                        keyw.lang = value['lang']
                    except KeyError:
                        pass
                else:
                    raise SAMLError('Configuration error: ui_info keywords')
                inst.append(keyw)
        elif isinstance(val, dict):
            keyw = mdui.Keywords()
            keyw.text = ' '.join(val['text'])
            try:
                keyw.lang = val['lang']
            except KeyError:
                pass
            inst.append(keyw)
        else:
            raise SAMLError('Configuration Error: ui_info keywords')
    except KeyError:
        pass
    return uii