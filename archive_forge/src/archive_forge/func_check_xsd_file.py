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
def check_xsd_file(self):
    if expected_errors > 0:
        schema = schema_class(xsd_file, validation='lax', locations=locations, defuse=defuse, loglevel=loglevel)
    else:
        schema = schema_class(xsd_file, locations=locations, defuse=defuse, loglevel=loglevel)
    self.errors.extend(schema.maps.all_errors)
    if inspect:
        components_ids = set([id(c) for c in schema.maps.iter_components()])
        components_ids.update((id(c) for c in schema.meta_schema.iter_components()))
        missing = [c for c in SchemaObserver.components if id(c) not in components_ids]
        if missing:
            raise ValueError('schema missing %d components: %r' % (len(missing), missing))
    if not inspect and (not no_pickle):
        try:
            obj = pickle.dumps(schema)
            deserialized_schema = pickle.loads(obj)
        except pickle.PicklingError:
            for e in schema.maps.iter_components():
                elem = getattr(e, 'elem', getattr(e, 'root', None))
                if isinstance(elem, PyElementTree.Element):
                    break
            else:
                raise
        else:
            self.assertTrue(isinstance(deserialized_schema, XMLSchemaBase), msg=xsd_file)
            self.assertEqual(schema.built, deserialized_schema.built, msg=xsd_file)
    if not inspect and (not self.errors):
        xpath_root = schema.xpath_node
        element_nodes = [x for x in xpath_root.iter() if hasattr(x, 'elem')]
        descendants = [x for x in xpath_root.iter_descendants('descendant-or-self')]
        self.assertTrue((x in descendants for x in element_nodes))
        context_xsd_elements = [e.value for e in element_nodes]
        for xsd_element in schema.iter():
            self.assertIn(xsd_element, context_xsd_elements, msg=xsd_file)
    for xsd_type in schema.maps.iter_components(xsd_classes=XsdType):
        self.assertIn(xsd_type.content_type_label, {'empty', 'simple', 'element-only', 'mixed'}, msg=xsd_file)
    if not expected_errors and schema_class.XSD_VERSION == '1.0':
        try:
            XMLSchema11(xsd_file, locations=locations, defuse=defuse, loglevel=loglevel)
        except XMLSchemaParseError as err:
            if not isinstance(err.validator, Xsd11ComplexType) or 'is simple or has a simple content' not in str(err):
                raise
            schema = schema_class(xsd_file, validation='lax', locations=locations, defuse=defuse, loglevel=loglevel)
            for error in schema.all_errors:
                if not isinstance(err.validator, Xsd11ComplexType) or 'is simple or has a simple content' not in str(err):
                    raise error
    if codegen and PythonGenerator is not None and (not self.errors) and all(('schemaLocation' in e.attrib for e in schema.root if e.tag == XSD_IMPORT)):
        generator = PythonGenerator(schema)
        with tempfile.TemporaryDirectory() as tempdir:
            cwd = os.getcwd()
            try:
                schema.export(tempdir, save_remote=True)
                os.chdir(tempdir)
                generator.render_to_files('bindings.py.jinja')
                spec = importlib.util.spec_from_file_location(tempdir, 'bindings.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            finally:
                os.chdir(cwd)