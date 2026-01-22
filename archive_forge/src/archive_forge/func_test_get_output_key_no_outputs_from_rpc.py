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
def test_get_output_key_no_outputs_from_rpc(self):
    self.parent_resource.nested_identifier = mock.Mock()
    self.parent_resource.nested_identifier.return_value = {'foo': 'bar'}
    self.parent_resource._rpc_client = mock.MagicMock()
    output = {}
    self.parent_resource._rpc_client.show_stack.return_value = [output]
    self.assertRaises(exception.NotFound, self.parent_resource.get_output, 'key')