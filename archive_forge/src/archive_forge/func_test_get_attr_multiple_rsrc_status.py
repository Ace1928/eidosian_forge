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
def test_get_attr_multiple_rsrc_status(self):
    """Test resolution of get_attr occurrences in HOT template."""
    hot_tpl = hot_tpl_generic_resource
    self.stack = parser.Stack(self.ctx, 'test_get_attr', template.Template(hot_tpl))
    self.stack.store()
    with mock.patch.object(rsrc_defn.ResourceDefinition, 'dep_attrs') as mock_da:
        mock_da.return_value = ['foo']
        self.stack.create()
    self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
    snippet = {'Value': {'get_attr': ['resource1', 'foo']}}
    rsrc = self.stack['resource1']
    for action, status in ((rsrc.CREATE, rsrc.IN_PROGRESS), (rsrc.CREATE, rsrc.COMPLETE), (rsrc.RESUME, rsrc.IN_PROGRESS), (rsrc.RESUME, rsrc.COMPLETE), (rsrc.UPDATE, rsrc.IN_PROGRESS), (rsrc.UPDATE, rsrc.COMPLETE)):
        rsrc.state_set(action, status)
        self.assertEqual({'Value': 'resource1'}, self.resolve(snippet))