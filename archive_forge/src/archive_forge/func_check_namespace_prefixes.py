import unittest
import re
import os
from textwrap import dedent
from xml.etree.ElementTree import Element, iselement
from xmlschema.exceptions import XMLSchemaValueError
from xmlschema.names import XSD_NAMESPACE, XSI_NAMESPACE, XSD_SCHEMA
from xmlschema.helpers import get_namespace
from xmlschema.resources import fetch_namespaces
from xmlschema.validators import XMLSchema10
from ._helpers import etree_elements_assert_equal
def check_namespace_prefixes(self, s):
    """Checks that a string doesn't contain protected prefixes (ns0, ns1 ...)."""
    match = PROTECTED_PREFIX_PATTERN.search(s)
    if match:
        msg = 'Protected prefix {!r} found:\n {}'.format(match.group(0), s)
        self.assertIsNone(match, msg)