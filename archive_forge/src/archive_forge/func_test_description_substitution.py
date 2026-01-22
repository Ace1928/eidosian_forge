from unittest import mock
from oslo_serialization import jsonutils
from heat.common import exception
from heat.engine import constraints
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import support
from heat.engine import translation
from heat.tests import common
def test_description_substitution(self):
    schema = {'description': properties.Schema('String', update_allowed=True), 'not_description': properties.Schema('String', update_allowed=True)}
    blank_rsrc = rsrc_defn.ResourceDefinition('foo', 'FooResource', {}, description='Foo resource')
    bar_rsrc = rsrc_defn.ResourceDefinition('foo', 'FooResource', {'description': 'bar'}, description='Foo resource')
    blank_props = blank_rsrc.properties(schema)
    self.assertEqual('Foo resource', blank_props['description'])
    self.assertEqual(None, blank_props['not_description'])
    replace_schema = {'description': properties.Schema('String')}
    empty_props = blank_rsrc.properties(replace_schema)
    self.assertEqual(None, empty_props['description'])
    bar_props = bar_rsrc.properties(schema)
    self.assertEqual('bar', bar_props['description'])