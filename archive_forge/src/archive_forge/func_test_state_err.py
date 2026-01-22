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
def test_state_err(self):
    """Test case when check_create_complete should raise error.

        check_create_complete should raise error when create task is
        done but the nested stack is not in (<action>,COMPLETE) state
        """
    self.status[1] = 'FAILED'
    reason = 'Resource %s failed: ValueError: resources.%s: broken on purpose' % (self.action.upper(), 'child_res')
    exp_path = 'resources.test.resources.child_res'
    exp = 'ValueError: %s: broken on purpose' % exp_path
    self.status[2] = reason
    complete = getattr(self.parent_resource, 'check_%s_complete' % self.action)
    exc = self.assertRaises(exception.ResourceFailure, complete, None)
    self.assertEqual(exp, str(exc))
    self.mock_status.assert_called_once_with(self.parent_resource.context, self.parent_resource.resource_id)