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
def check_decode_to_objects(self, root, with_bindings=False):
    data_element = self.schema.to_objects(xml_file, with_bindings)
    self.assertIsInstance(data_element, DataElement)
    self.assertEqual(data_element.tag, root.tag)
    if not with_bindings:
        self.assertIs(data_element.__class__, DataElement)
    else:
        self.assertEqual(data_element.tag, root.tag)
        self.assertTrue(data_element.__class__.__name__.endswith('Binding'))