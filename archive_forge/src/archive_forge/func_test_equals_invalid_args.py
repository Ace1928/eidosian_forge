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
def test_equals_invalid_args(self):
    tmpl = template.Template(aws_empty_template)
    snippet = {'Fn::Equals': ['test', 'prod', 'invalid']}
    exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
    error_msg = '.Fn::Equals: Arguments to "Fn::Equals" must be of the form: [value_1, value_2]'
    self.assertIn(error_msg, str(exc))
    snippet = {'Fn::Equals': {'equal': False}}
    exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
    self.assertIn(error_msg, str(exc))