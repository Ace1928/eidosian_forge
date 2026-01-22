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
def verify_create_params(self, wf_yaml):
    wf = yaml.safe_load(wf_yaml)['create_vm']
    self.assertEqual(['on_error'], wf['task-defaults']['on-error'])
    tasks = wf['tasks']
    task = tasks['wait_instance']
    self.assertEqual('vm_id_new in <% $.list_servers %>', task['with-items'])
    self.assertEqual(5, task['retry']['delay'])
    self.assertEqual(15, task['retry']['count'])
    self.assertEqual(8, task['wait-after'])
    self.assertTrue(task['pause-before'])
    self.assertEqual(11, task['timeout'])
    self.assertEqual('test', task['target'])
    self.assertEqual(7, task['wait-before'])
    self.assertFalse(task['keep-result'])
    return [FakeWorkflow('create_vm')]