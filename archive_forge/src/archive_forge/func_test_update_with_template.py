import contextlib
import json
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import exceptions as msg_exceptions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources import stack_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_update_with_template(self):
    if self.adopt_data is not None:
        return
    ident = identifier.HeatIdentifier(self.ctx.tenant_id, 'fake_name', 'pancakes')
    self.parent_resource.resource_id = ident.stack_id
    self.parent_resource.nested_identifier = mock.Mock(return_value=ident)
    self.parent_resource.child_params = mock.Mock(return_value=self.params)
    rpcc = mock.Mock()
    self.parent_resource.rpc_client = rpcc
    rpcc.return_value._update_stack.return_value = dict(ident)
    status = ('CREATE', 'COMPLETE', '', 'now_time')
    with self.patchobject(stack_object.Stack, 'get_status', return_value=status):
        self.parent_resource.update_with_template(self.empty_temp, user_params=self.params, timeout_mins=self.timeout_mins)
    rpcc.return_value._update_stack.assert_called_once_with(self.ctx, stack_identity=dict(ident), template_id=self.IntegerMatch(), template=None, params=None, files=None, args={rpc_api.PARAM_TIMEOUT: self.timeout_mins, rpc_api.PARAM_CONVERGE: False})