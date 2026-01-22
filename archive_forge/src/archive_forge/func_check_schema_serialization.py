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
def check_schema_serialization(self):
    serialized_schema = pickle.dumps(self.schema)
    deserialized_schema = pickle.loads(serialized_schema)
    deserialized_errors = []
    deserialized_chunks = []
    for obj in deserialized_schema.iter_decode(xml_file):
        if isinstance(obj, xmlschema.XMLSchemaValidationError):
            deserialized_errors.append(obj)
        else:
            deserialized_chunks.append(obj)
    self.assertEqual(len(deserialized_errors), len(self.errors), msg=xml_file)
    self.assertEqual(deserialized_chunks, self.chunks, msg=xml_file)