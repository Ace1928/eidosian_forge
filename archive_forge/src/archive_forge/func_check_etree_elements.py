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
def check_etree_elements(self, elem, other):
    """Checks if two ElementTree elements are equal."""
    try:
        self.assertIsNone(etree_elements_assert_equal(elem, other, strict=False, skip_comments=True))
    except AssertionError as err:
        self.assertIsNone(err, None)