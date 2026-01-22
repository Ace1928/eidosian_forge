from importlib import import_module
import logging
import os
import sys
from saml2 import NAMESPACE
from saml2 import ExtensionElement
from saml2 import SAMLError
from saml2 import extension_elements_to_elements
from saml2 import saml
from saml2.s_utils import do_ava
from saml2.s_utils import factory
from saml2.saml import NAME_FORMAT_UNSPECIFIED
from saml2.saml import NAMEID_FORMAT_PERSISTENT
def ava_from(self, attribute, allow_unknown=False):
    try:
        attr = self._fro[attribute.name.strip().lower()]
    except AttributeError:
        attr = attribute.friendly_name.strip().lower()
    except KeyError:
        if allow_unknown:
            try:
                attr = attribute.name.strip().lower()
            except AttributeError:
                attr = attribute.friendly_name.strip().lower()
        else:
            raise
    val = []
    for value in attribute.attribute_value:
        if value.extension_elements:
            ext = extension_elements_to_elements(value.extension_elements, [saml])
            for ex in ext:
                if attr == 'eduPersonTargetedID' and ex.text:
                    val.append(ex.text.strip())
                else:
                    cval = {}
                    for key, (name, typ, mul) in ex.c_attributes.items():
                        exv = getattr(ex, name)
                        if exv:
                            cval[name] = exv
                    if ex.text:
                        cval['value'] = ex.text.strip()
                    val.append({ex.c_tag: cval})
        elif not value.text:
            val.append('')
        else:
            val.append(value.text.strip())
    return (attr, val)