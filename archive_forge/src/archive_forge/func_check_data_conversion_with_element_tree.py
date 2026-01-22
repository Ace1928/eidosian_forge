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
def check_data_conversion_with_element_tree(self):
    root = ElementTree.parse(xml_file).getroot()
    namespaces = fetch_namespaces(xml_file)
    options = {'namespaces': namespaces}
    self.check_decode_encode(root, cdata_prefix='#', **options)
    self.check_decode_encode(root, UnorderedConverter, cdata_prefix='#', **options)
    self.check_decode_encode(root, ParkerConverter, validation='lax', **options)
    self.check_decode_encode(root, ParkerConverter, validation='skip', **options)
    self.check_decode_encode(root, BadgerFishConverter, **options)
    self.check_decode_encode(root, AbderaConverter, **options)
    self.check_decode_encode(root, JsonMLConverter, **options)
    self.check_decode_encode(root, ColumnarConverter, validation='lax', **options)
    self.check_decode_encode(root, DataElementConverter, **options)
    self.check_decode_encode(root, DataBindingConverter, **options)
    self.schema.maps.clear_bindings()
    self.check_json_serialization(root, cdata_prefix='#', **options)
    self.check_json_serialization(root, UnorderedConverter, **options)
    self.check_json_serialization(root, ParkerConverter, validation='lax', **options)
    self.check_json_serialization(root, ParkerConverter, validation='skip', **options)
    self.check_json_serialization(root, BadgerFishConverter, **options)
    self.check_json_serialization(root, AbderaConverter, **options)
    self.check_json_serialization(root, JsonMLConverter, **options)
    self.check_json_serialization(root, ColumnarConverter, validation='lax', **options)
    self.check_decode_to_objects(root)
    self.check_decode_to_objects(root, with_bindings=True)
    self.schema.maps.clear_bindings()