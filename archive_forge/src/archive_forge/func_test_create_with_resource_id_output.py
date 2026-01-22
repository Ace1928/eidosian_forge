from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as client
from heat.engine import resource
from heat.engine.resources.openstack.mistral import external_resource
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_create_with_resource_id_output(self):
    output = '{"resource_id": "my-fake-resource-id"}'
    execution = self._create_resource('execution', self.rsrc_defn, self.stack, output)
    scheduler.TaskRunner(execution.create)()
    expected_state = (execution.CREATE, execution.COMPLETE)
    self.assertEqual(expected_state, execution.state)
    self.assertEqual('my-fake-resource-id', execution.resource_id)