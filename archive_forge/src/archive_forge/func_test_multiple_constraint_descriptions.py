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
def test_multiple_constraint_descriptions(self):
    len_desc = 'string length should be between 8 and 16'
    pattern_desc1 = 'Value must consist of characters only'
    pattern_desc2 = 'Value must start with a lowercase character'
    param = {'db_name': {'description': 'The WordPress database name', 'type': 'string', 'default': 'wordpress', 'constraints': [{'length': {'min': 6, 'max': 16}, 'description': len_desc}, {'allowed_pattern': '[a-zA-Z]+', 'description': pattern_desc1}, {'allowed_pattern': '[a-z]+[a-zA-Z]*', 'description': pattern_desc2}]}}
    name = 'db_name'
    schema = param['db_name']

    def v(value):
        param_schema = hot_param.HOTParamSchema.from_dict(name, schema)
        param_schema.validate()
        param_schema.validate_value(value)
        return True
    value = 'wp'
    err = self.assertRaises(exception.StackValidationFailed, v, value)
    self.assertIn(len_desc, str(err))
    value = 'abcdefghijklmnopq'
    err = self.assertRaises(exception.StackValidationFailed, v, value)
    self.assertIn(len_desc, str(err))
    value = 'abcdefgh1'
    err = self.assertRaises(exception.StackValidationFailed, v, value)
    self.assertIn(pattern_desc1, str(err))
    value = 'Abcdefghi'
    err = self.assertRaises(exception.StackValidationFailed, v, value)
    self.assertIn(pattern_desc2, str(err))
    value = 'abcdefghi'
    self.assertTrue(v(value))
    value = 'abcdefghI'
    self.assertTrue(v(value))