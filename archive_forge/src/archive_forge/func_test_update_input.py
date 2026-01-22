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
def test_update_input(self):
    wf = self._create_resource('workflow')
    t = template_format.parse(workflow_template)
    t['resources']['workflow']['properties']['input'] = {'foo': 'bar'}
    rsrc_defns = template.Template(t).resource_definitions(self.stack)
    new_wf = rsrc_defns['workflow']
    self.mistral.workflows.update.return_value = [FakeWorkflow('test_stack-workflow-b5fiekfci3yc')]
    scheduler.TaskRunner(wf.update, new_wf)()
    self.assertTrue(self.mistral.workflows.update.called)
    self.assertEqual((wf.UPDATE, wf.COMPLETE), wf.state)