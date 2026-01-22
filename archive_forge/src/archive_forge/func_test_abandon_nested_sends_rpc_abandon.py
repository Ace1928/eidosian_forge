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
def test_abandon_nested_sends_rpc_abandon(self):
    rpcc = mock.MagicMock()

    @contextlib.contextmanager
    def exc_filter(*args):
        try:
            yield
        except exception.NotFound:
            pass
    rpcc.ignore_error_by_name.side_effect = exc_filter
    self.parent_resource.rpc_client = rpcc
    self.parent_resource.resource_id = 'fake_id'
    self.parent_resource.prepare_abandon()
    status = ('CREATE', 'COMPLETE', '', 'now_time')
    with mock.patch.object(stack_object.Stack, 'get_status', return_value=status):
        self.parent_resource.delete_nested()
    rpcc.return_value.abandon_stack.assert_called_once_with(self.parent_resource.context, mock.ANY)
    rpcc.return_value.delete_stack.assert_not_called()