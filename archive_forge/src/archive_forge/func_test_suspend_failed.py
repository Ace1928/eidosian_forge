import collections
import json
from unittest import mock
from heatclient import exc
from heatclient.v1 import stacks
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import remote_stack
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common as tests_common
from heat.tests import utils
def test_suspend_failed(self):
    stacks = [get_stack(stack_status='SUSPEND_IN_PROGRESS'), get_stack(stack_status='SUSPEND_FAILED', stack_status_reason='Remote stack suspend failed')]
    rsrc = self.create_remote_stack()
    self.heat.stacks.get = mock.MagicMock(side_effect=stacks)
    self.heat.actions.suspend = mock.MagicMock()
    error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.suspend))
    error_msg = 'ResourceInError: resources.remote_stack: Went to status SUSPEND_FAILED due to "Remote stack suspend failed"'
    self.assertEqual(error_msg, str(error))
    self.assertEqual((rsrc.SUSPEND, rsrc.FAILED), rsrc.state)
    self.heat.actions.suspend.assert_has_calls([])