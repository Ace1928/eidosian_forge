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
def test_depends_condition(self):
    hot_tpl = template_format.parse('\n        heat_template_version: 2016-10-14\n        resources:\n          one:\n            type: OS::Heat::None\n          two:\n            type: OS::Heat::None\n            condition: False\n          three:\n            type: OS::Heat::None\n            depends_on: two\n        ')
    tmpl = template.Template(hot_tpl)
    stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
    stack.validate()
    self.assertEqual({'one', 'three'}, set(stack.resources))