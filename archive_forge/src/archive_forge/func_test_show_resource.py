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
def test_show_resource(self):
    self._create_stack()
    self.l7policy.resource_id_set('1234')
    self.octavia_client.l7policy_show.return_value = {'id': '1234'}
    self.assertEqual({'id': '1234'}, self.l7policy._show_resource())
    self.octavia_client.l7policy_show.assert_called_with('1234')