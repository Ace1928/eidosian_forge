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
def test_need_update_check_failed_state_after_mark_unhealthy(self):
    self.parent_resource.resource_id = 'fake_id'
    self.parent_resource.state_set(self.parent_resource.CHECK, self.parent_resource.FAILED)
    self.nested = mock.MagicMock()
    self.nested.name = 'nested-stack'
    self.parent_resource.nested = mock.MagicMock(return_value=self.nested)
    self.parent_resource._nested = self.nested
    self.parent_resource._rpc_client = mock.MagicMock()
    self.parent_resource._rpc_client.show_stack.return_value = [{'stack_action': self.parent_resource.CREATE, 'stack_status': self.parent_resource.COMPLETE}]
    self.assertRaises(resource.UpdateReplace, self.parent_resource._needs_update, self.parent_resource.t, self.parent_resource.t, self.parent_resource.properties, self.parent_resource.properties, self.parent_resource)