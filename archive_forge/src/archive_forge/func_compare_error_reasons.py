import pdb
import os
import ast
import pickle
import re
import time
import logging
import importlib
import tempfile
import warnings
from xml.etree import ElementTree
from elementpath.etree import PyElementTree, etree_tostring
import xmlschema
from xmlschema import XMLSchemaBase, XMLSchema11, XMLSchemaValidationError, \
from xmlschema.names import XSD_IMPORT
from xmlschema.helpers import local_name
from xmlschema.resources import fetch_namespaces
from xmlschema.validators import XsdType, Xsd11ComplexType
from xmlschema.dataobjects import DataElementConverter, DataBindingConverter, DataElement
from ._helpers import iter_nested_items, etree_elements_assert_equal
from ._case_class import XsdValidatorTestCase
from ._observers import SchemaObserver
def compare_error_reasons(reason, other_reason):
    if ' at 0x' in reason:
        self.assertEqual(OBJ_ID_PATTERN.sub(' at 0xff', reason), OBJ_ID_PATTERN.sub(' at 0xff', other_reason), msg=xml_file)
    else:
        self.assertEqual(reason, other_reason, msg=xml_file)