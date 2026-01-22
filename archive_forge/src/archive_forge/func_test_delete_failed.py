from unittest import mock
import yaml
from osc_lib import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.resources.openstack.octavia import l7policy
from heat.tests import common
from heat.tests.openstack.octavia import inline_templates
from heat.tests import utils
def test_delete_failed(self):
    self._create_stack()
    self.l7policy.resource_id_set('1234')
    self.octavia_client.l7policy_delete.side_effect = exceptions.Unauthorized(401)
    self.l7policy.handle_delete()
    self.assertRaises(exceptions.Unauthorized, self.l7policy.check_delete_complete, None)
    self.octavia_client.l7policy_delete.assert_called_with('1234')