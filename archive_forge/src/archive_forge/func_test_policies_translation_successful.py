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
def test_policies_translation_successful(self):
    tmpl = template_format.parse(workflow_template_policies_translation)
    stack = utils.parse_stack(tmpl)
    rsrc_defns = stack.t.resource_definitions(stack)['workflow']
    wf = workflow.Workflow('workflow', rsrc_defns, stack)
    result = {k: v for k, v in wf.properties['tasks'][0].items() if v}
    self.assertEqual({'name': 'check_dat_thing', 'action': 'nova.servers_list', 'retry': {'delay': 5, 'count': 15}, 'wait_before': 5, 'wait_after': 5, 'pause_before': True, 'timeout': 42, 'concurrency': 5}, result)