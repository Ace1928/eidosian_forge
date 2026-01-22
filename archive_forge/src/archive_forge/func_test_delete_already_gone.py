from unittest import mock
import yaml
from osc_lib import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.resources.openstack.octavia import l7rule
from heat.tests import common
from heat.tests.openstack.octavia import inline_templates
from heat.tests import utils
def test_delete_already_gone(self):
    self._create_stack()
    self.l7rule.resource_id_set('1234')
    self.octavia_client.l7rule_delete.side_effect = exceptions.NotFound(404)
    self.l7rule.handle_delete()
    self.assertTrue(self.l7rule.check_delete_complete(None))