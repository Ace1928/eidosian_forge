import copy
import json
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_yaql(self):
    param1 = -800
    param2 = [-8, 0, 4, -11, 2]
    env = environment.Environment({'parameters': {'param1': param1, 'param2': param2}})
    stack = self.create_stack(self.template_yaql, env)
    my_value = stack['my_value']
    self.assertEqual(param1, my_value.FnGetAtt('value'))
    my_value2 = stack['my_value2']
    self.assertEqual(min(param2), my_value2.FnGetAtt('value'))
    my_value3 = stack['my_value3']
    self.assertEqual(param1, my_value3.FnGetAtt('value'))