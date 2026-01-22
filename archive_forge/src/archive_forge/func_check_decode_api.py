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
def check_decode_api(self):
    strict_decoded_data = self.schema.decode(xml_file)
    lax_decoded_data = self.schema.decode(xml_file, validation='lax')
    skip_decoded_data = self.schema.decode(xml_file, validation='skip')
    self.assertEqual(strict_decoded_data, self.chunks[0], msg=xml_file)
    self.assertEqual(lax_decoded_data[0], self.chunks[0], msg=xml_file)
    self.assertEqual(skip_decoded_data, self.chunks[0], msg=xml_file)