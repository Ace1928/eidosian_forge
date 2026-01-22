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
def test_xml_document_validation(self):
    if not validation_only:
        self.check_decoding_with_element_tree()
        if not inspect and (not no_pickle):
            self.check_schema_serialization()
        if not self.errors:
            self.check_data_conversion_with_element_tree()
        if lxml_etree is not None:
            self.check_data_conversion_with_lxml()
    self.check_iter_errors()
    self.check_validate_and_is_valid_api()
    if check_with_lxml and lxml_etree is not None:
        self.check_lxml_validation()
    if not validation_only and codegen and (PythonGenerator is not None) and (not self.errors) and (not self.schema.all_errors) and all(('schemaLocation' in e.attrib for e in self.schema.root if e.tag == XSD_IMPORT)):
        self.check_validation_with_generated_code()