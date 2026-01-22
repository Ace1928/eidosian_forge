from unittest import mock
import yaml
from mistralclient.api import base as mistral_base
from mistralclient.api.v2 import executions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as client
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.mistral import workflow
from heat.engine.resources import signal_responder
from heat.engine.resources import stack_user
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_update_failed_replace_not_found(self):
    wf = self._create_resource('workflow', workflow_template_update_replace)
    t = template_format.parse(workflow_template_update_replace_failed)
    rsrc_defns = template.Template(t).resource_definitions(self.stack)
    new_wf = rsrc_defns['workflow']
    self.mistral.workflows.update.side_effect = Exception('boom!')
    self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(wf.update, new_wf))
    self.assertEqual((wf.UPDATE, wf.FAILED), wf.state)
    self.mistral.workflows.get.side_effect = [mistral_base.APIException(error_code=404)]
    self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(wf.update, new_wf))