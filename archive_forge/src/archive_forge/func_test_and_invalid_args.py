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
def test_and_invalid_args(self):
    tmpl = template.Template(aws_empty_template)
    error_msg = 'The minimum number of condition arguments to "Fn::And" is 2.'
    snippet = {'Fn::And': ['invalid_arg']}
    exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
    self.assertIn(error_msg, str(exc))
    error_msg = 'Arguments to "Fn::And" must be'
    snippet = {'Fn::And': 'invalid'}
    exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
    self.assertIn(error_msg, str(exc))
    stk = stack.Stack(utils.dummy_context(), 'test_and_invalid', tmpl)
    snippet = {'Fn::And': ['cd1', True]}
    exc = self.assertRaises(ValueError, self.resolve_condition, snippet, tmpl, stk)
    error_msg = 'Invalid condition "cd1"'
    self.assertIn(error_msg, str(exc))