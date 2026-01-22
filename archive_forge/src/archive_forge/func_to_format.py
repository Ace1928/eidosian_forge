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
def to_format(self, attr):
    """Creates an Attribute instance with name, name_format and
        friendly_name

        :param attr: The local name of the attribute
        :return: An Attribute instance
        """
    try:
        _attr = self._to[attr]
    except KeyError:
        try:
            _attr = self._to[attr.lower()]
        except KeyError:
            _attr = ''
    if _attr:
        return factory(saml.Attribute, name=_attr, name_format=self.name_format, friendly_name=attr)
    else:
        return factory(saml.Attribute, name=attr)