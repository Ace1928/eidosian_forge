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
def test_stack_describe_bad_tenant(self):
    non_exist_identifier = identifier.HeatIdentifier('wibble', 'wibble', '18d06e2e-44d3-4bef-9fbf-52480d604b02')
    invalid_tenant_exc = exception.InvalidTenant(target='test', actual='test')
    self.patchobject(service.EngineService, '_get_stack', side_effect=invalid_tenant_exc)
    ex = self.assertRaises(dispatcher.ExpectedException, self.eng.show_stack, self.ctx, non_exist_identifier, resolve_outputs=True)
    self.assertEqual(exception.InvalidTenant, ex.exc_info[0])
    service.EngineService._get_stack.assert_called_once_with(self.ctx, non_exist_identifier, show_deleted=True)