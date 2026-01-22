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
def list_to_local(acs, attrlist, allow_unknown_attributes=False):
    """Replaces the attribute names in a attribute value assertion with the
    equivalent name from a local name format.

    :param acs: List of Attribute Converters
    :param attrlist: List of Attributes
    :param allow_unknown_attributes: If unknown attributes are allowed
    :return: A key,values dictionary
    """
    if not acs:
        acs = [AttributeConverter()]
        acsd = {'': acs}
    else:
        acsd = {a.name_format: a for a in acs}
    ava = {}
    for attr in attrlist:
        try:
            _func = acsd[attr.name_format].ava_from
        except KeyError:
            if attr.name_format == NAME_FORMAT_UNSPECIFIED or allow_unknown_attributes:
                _func = acs[0].lcd_ava_from
            else:
                logger.info('Unsupported attribute name format: %s', attr.name_format)
                continue
        try:
            key, val = _func(attr)
        except KeyError:
            if allow_unknown_attributes:
                key, val = acs[0].lcd_ava_from(attr)
            else:
                logger.info('Unknown attribute name: %s', attr)
                continue
        except AttributeError:
            continue
        try:
            ava[key].extend(val)
        except KeyError:
            ava[key] = val
    return ava