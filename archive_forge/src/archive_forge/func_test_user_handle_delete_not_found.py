from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import user
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_user_handle_delete_not_found(self):
    exc = self.keystoneclient.NotFound
    self.users.delete.side_effect = exc
    self.assertIsNone(self.test_user.handle_delete())