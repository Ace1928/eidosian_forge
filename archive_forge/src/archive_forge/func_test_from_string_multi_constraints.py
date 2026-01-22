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
def test_from_string_multi_constraints(self):
    description = 'WebServer EC2 instance type'
    allowed_pattern = '[A-Za-z0-9.]*'
    constraint_desc = 'Must contain only alphanumeric characters.'
    param = parameters.Schema.from_dict('name', {'Type': 'String', 'Description': description, 'Default': 'm1.large', 'MinLength': '7', 'AllowedPattern': allowed_pattern, 'ConstraintDescription': constraint_desc})
    schema = properties.Schema.from_parameter(param)
    self.assertEqual(properties.Schema.STRING, schema.type)
    self.assertEqual(description, schema.description)
    self.assertIsNone(schema.default)
    self.assertFalse(schema.required)
    self.assertEqual(2, len(schema.constraints))
    len_constraint = schema.constraints[0]
    allowed_constraint = schema.constraints[1]
    self.assertEqual(7, len_constraint.min)
    self.assertIsNone(len_constraint.max)
    self.assertEqual(allowed_pattern, allowed_constraint.pattern)
    self.assertEqual(constraint_desc, allowed_constraint.description)
    props = properties.Properties({'test': schema}, {})
    props.validate()