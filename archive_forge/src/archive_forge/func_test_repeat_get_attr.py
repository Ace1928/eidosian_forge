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
def test_repeat_get_attr(self):
    """Test repeat function with get_attr function as an argument."""
    tmpl = template.Template(hot_tpl_complex_attrs_all_attrs)
    self.stack = parser.Stack(self.ctx, 'test_repeat_get_attr', tmpl)
    snippet = {'repeat': {'template': 'this is %var%', 'for_each': {'%var%': {'get_attr': ['resource1', 'list']}}}}
    repeat = self.stack.t.parse(self.stack.defn, snippet)
    self.stack.store()
    with mock.patch.object(rsrc_defn.ResourceDefinition, 'dep_attrs') as mock_da:
        mock_da.return_value = ['list']
        self.stack.create()
    self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
    self.assertEqual(['this is foo', 'this is bar'], function.resolve(repeat))