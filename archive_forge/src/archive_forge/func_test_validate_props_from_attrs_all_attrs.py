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
def test_validate_props_from_attrs_all_attrs(self):
    stack = parser.Stack(self.ctx, 'test_props_from_attrs', template.Template(hot_tpl_mapped_props_all_attrs))
    stack.resources['resource1'].list = None
    stack.resources['resource1'].map = None
    stack.resources['resource1'].string = None
    try:
        stack.validate()
    except exception.StackValidationFailed as exc:
        self.fail('Validation should have passed: %s' % str(exc))
    self.assertEqual([], stack.resources['resource2'].properties['a_list'])
    self.assertEqual({}, stack.resources['resource2'].properties['a_map'])
    self.assertEqual('', stack.resources['resource2'].properties['a_string'])