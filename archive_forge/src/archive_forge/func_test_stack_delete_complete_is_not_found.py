from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from heat.common import context
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.engine.cfn import template as cfntemplate
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import template as hottemplate
from heat.engine import resource as res
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_stack_delete_complete_is_not_found(self):
    t = template_format.parse(tools.wp_template)
    tmpl = templatem.Template(t)
    stack = parser.Stack(self.ctx, 'delete_complete_stack', tmpl)
    self.patchobject(self.eng, '_get_stack')
    self.patchobject(parser.Stack, 'load', return_value=stack)
    stack.status = stack.COMPLETE
    stack.action = stack.DELETE
    stack.convergence = True
    self.eng.thread_group_mgr.start = mock.MagicMock()
    ex = self.assertRaises(dispatcher.ExpectedException, self.eng.delete_stack, 'irrelevant', 'irrelevant')
    self.assertEqual(exception.EntityNotFound, ex.exc_info[0])
    self.eng.thread_group_mgr.start.assert_called_once_with(None, stack.purge_db)