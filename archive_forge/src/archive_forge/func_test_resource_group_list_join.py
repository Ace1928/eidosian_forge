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
def test_resource_group_list_join(self):
    """Test list_join on a ResourceGroup's inner attributes

        This should not fail during validation (i.e. before the ResourceGroup
        can return the list of the runtime values.
        """
    hot_tpl = template_format.parse('\n        heat_template_version: 2014-10-16\n        resources:\n          rg:\n            type: OS::Heat::ResourceGroup\n            properties:\n              count: 3\n              resource_def:\n                type: OS::Nova::Server\n        ')
    tmpl = template.Template(hot_tpl)
    stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
    snippet = {'list_join': ['\n', {'get_attr': ['rg', 'name']}]}
    self.assertEqual('', self.resolve(snippet, tmpl, stack))
    hot_tpl['heat_template_version'] = '2015-10-15'
    tmpl = template.Template(hot_tpl)
    stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
    snippet = {'list_join': ['\n', {'get_attr': ['rg', 'name']}]}
    self.assertEqual('', self.resolve(snippet, tmpl, stack))
    snippet = {'list_join': ['\n', {'get_attr': ['rg', 'name']}, {'get_attr': ['rg', 'name']}]}
    self.assertEqual('', self.resolve(snippet, tmpl, stack))