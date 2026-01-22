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
def test_range_constraint(self):
    range_desc = 'Value must be between 30000 and 50000'
    param = {'db_port': {'description': 'The database port', 'type': 'number', 'default': 31000, 'constraints': [{'range': {'min': 30000, 'max': 50000}, 'description': range_desc}]}}
    name = 'db_port'
    schema = param['db_port']

    def v(value):
        param_schema = hot_param.HOTParamSchema.from_dict(name, schema)
        param_schema.validate()
        param_schema.validate_value(value)
        return True
    value = 29999
    err = self.assertRaises(exception.StackValidationFailed, v, value)
    self.assertIn(range_desc, str(err))
    value = 50001
    err = self.assertRaises(exception.StackValidationFailed, v, value)
    self.assertIn(range_desc, str(err))
    value = 30000
    self.assertTrue(v(value))
    value = 40000
    self.assertTrue(v(value))
    value = 50000
    self.assertTrue(v(value))