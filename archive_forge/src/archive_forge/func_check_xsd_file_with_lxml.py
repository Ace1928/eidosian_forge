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
def check_xsd_file_with_lxml(self, xmlschema_time):
    start_time = time.time()
    lxs = lxml_etree.parse(xsd_file)
    try:
        lxml_etree.XMLSchema(lxs.getroot())
    except lxml_etree.XMLSchemaParseError as err:
        if not self.errors:
            print('\nSchema error with lxml.etree.XMLSchema for file {!r} ({}): {}'.format(xsd_file, self.__class__.__name__, str(err)))
    else:
        if self.errors:
            msg = '\nUnrecognized errors with lxml.etree.XMLSchema for file {!r} ({}): {}'
            print(msg.format(xsd_file, self.__class__.__name__, '\n++++++\n'.join([str(e) for e in self.errors])))
        lxml_schema_time = time.time() - start_time
        if lxml_schema_time >= xmlschema_time:
            msg = '\nSlower lxml.etree.XMLSchema ({:.3f}s VS {:.3f}s) with file {!r} ({})'
            print(msg.format(lxml_schema_time, xmlschema_time, xsd_file, self.__class__.__name__))