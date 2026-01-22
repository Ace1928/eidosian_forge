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
@classmethod
def casepath(cls, relative_path):
    """
        Returns the absolute path from a relative path specified from the referenced TEST_CASES_DIR.
        """
    return os.path.join(cls.TEST_CASES_DIR or '', relative_path)