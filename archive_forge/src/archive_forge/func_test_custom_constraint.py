import copy
from unittest import mock
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine.cfn import functions as cfn_functions
from heat.engine.cfn import parameters as cfn_param
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import parameters as hot_param
from heat.engine.hot import template as hot_template
from heat.engine import resource
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_custom_constraint(self):

    class ZeroConstraint(object):

        def validate(self, value, context):
            return value == '0'
    env = resources.global_env()
    env.register_constraint('zero', ZeroConstraint)
    self.addCleanup(env.constraints.pop, 'zero')
    desc = 'Value must be zero'
    param = {'param1': {'type': 'string', 'constraints': [{'custom_constraint': 'zero', 'description': desc}]}}
    name = 'param1'
    schema = param['param1']

    def v(value):
        param_schema = hot_param.HOTParamSchema.from_dict(name, schema)
        param_schema.validate()
        param_schema.validate_value(value)
        return True
    value = '1'
    err = self.assertRaises(exception.StackValidationFailed, v, value)
    self.assertEqual(desc, str(err))
    value = '2'
    err = self.assertRaises(exception.StackValidationFailed, v, value)
    self.assertEqual(desc, str(err))
    value = '0'
    self.assertTrue(v(value))