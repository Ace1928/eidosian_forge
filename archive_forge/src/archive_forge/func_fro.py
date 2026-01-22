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
def fro(self, statement):
    """Get the attributes and the attribute values.

        :param statement: The AttributeStatement.
        :return: A dictionary containing attributes and values
        """
    if not self.name_format:
        return self.fail_safe_fro(statement)
    result = {}
    for attribute in statement.attribute:
        if attribute.name_format and self.name_format and (attribute.name_format != self.name_format):
            continue
        try:
            key, val = self.ava_from(attribute)
        except (KeyError, AttributeError):
            pass
        else:
            result[key] = val
    return result