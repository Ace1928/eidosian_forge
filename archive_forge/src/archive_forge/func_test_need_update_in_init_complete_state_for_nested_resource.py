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
def test_need_update_in_init_complete_state_for_nested_resource(self):
    """Test the resource with no nested stack should need replacement.

        The resource in failed state and has no nested stack,
        should need update with UpdateReplace.
        """
    self.parent_resource.state_set(self.parent_resource.INIT, self.parent_resource.COMPLETE)
    self.parent_resource._nested = None
    self.assertRaises(resource.UpdateReplace, self.parent_resource._needs_update, self.parent_resource.t, self.parent_resource.t, self.parent_resource.properties, self.parent_resource.properties, self.parent_resource)