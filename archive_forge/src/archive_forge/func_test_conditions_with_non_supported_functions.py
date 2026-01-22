import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_conditions_with_non_supported_functions(self):
    t = {'heat_template_version': '2016-10-14', 'parameters': {'env_type': {'type': 'string', 'default': 'test'}}, 'conditions': {'prod_env': {'equals': [{'get_param': 'env_type'}, {'get_attr': [None, 'att']}]}}}
    tmpl = template.Template(t)
    stk = stack.Stack(self.ctx, 'test_condition_with_get_attr_func', tmpl)
    ex = self.assertRaises(exception.StackValidationFailed, tmpl.conditions, stk)
    self.assertIn('"get_attr" is invalid', str(ex))
    self.assertIn('conditions.prod_env.equals[1].get_attr', str(ex))
    tmpl.t['conditions']['prod_env'] = {'get_resource': 'R1'}
    stk = stack.Stack(self.ctx, 'test_condition_with_get_attr_func', tmpl)
    ex = self.assertRaises(exception.StackValidationFailed, tmpl.conditions, stk)
    self.assertIn('"get_resource" is invalid', str(ex))
    tmpl.t['conditions']['prod_env'] = {'get_attr': [None, 'att']}
    stk = stack.Stack(self.ctx, 'test_condition_with_get_attr_func', tmpl)
    ex = self.assertRaises(exception.StackValidationFailed, tmpl.conditions, stk)
    self.assertIn('"get_attr" is invalid', str(ex))