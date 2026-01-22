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
def test_not_invalid_args(self):
    tmpl = template.Template(aws_empty_template)
    stk = stack.Stack(utils.dummy_context(), 'test_not_invalid', tmpl)
    snippet = {'Fn::Not': ['invalid_arg']}
    exc = self.assertRaises(ValueError, self.resolve_condition, snippet, tmpl, stk)
    error_msg = 'Invalid condition "invalid_arg"'
    self.assertIn(error_msg, str(exc))
    snippet = {'Fn::Not': 'invalid'}
    exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
    error_msg = 'Arguments to "Fn::Not" must be '
    self.assertIn(error_msg, str(exc))
    snippet = {'Fn::Not': ['cd1', 'cd2']}
    exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
    error_msg = 'Arguments to "Fn::Not" must be '
    self.assertIn(error_msg, str(exc))