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
def d_to_local_name(acs, attr):
    """
    :param acs: List of AttributeConverter instances
    :param attr: an Attribute dictionary
    :return: The local attribute name
    """
    for aconv in acs:
        lattr = aconv.d_from_format(attr)
        if lattr:
            return lattr
    try:
        return attr['friendly_name']
    except KeyError:
        raise ConverterError(f'Could not find local name for {attr}')