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
def test_from_param_no_default(self):
    param = parameters.Schema.from_dict('name', {'Description': 'WebServer EC2 instance type', 'Type': 'String'})
    schema = properties.Schema.from_parameter(param)
    self.assertTrue(schema.required)
    self.assertIsNone(schema.default)
    self.assertEqual(0, len(schema.constraints))
    self.assertFalse(schema.allow_conversion)
    props = properties.Properties({'name': schema}, {'name': 'm1.large'})
    props.validate()