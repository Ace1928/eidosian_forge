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
def test_from_legacy_list(self):
    ls = properties.Schema.from_legacy({'Type': 'List', 'Default': ['wibble'], 'Schema': {'Type': 'String', 'Default': 'wibble', 'MaxLength': 8}})
    self.assertEqual(properties.Schema.LIST, ls.type)
    self.assertEqual(['wibble'], ls.default)
    ss = ls.schema[0]
    self.assertEqual(properties.Schema.STRING, ss.type)
    self.assertEqual('wibble', ss.default)