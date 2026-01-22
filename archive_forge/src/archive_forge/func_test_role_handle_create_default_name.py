import copy
from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import role
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_role_handle_create_default_name(self):
    test_role = self._get_rsrc(without_name=True)
    test_role.physical_resource_name = mock.Mock(return_value='phy_role_name')
    test_role.handle_create()
    self.roles.create.assert_called_once_with(name='phy_role_name', domain='default')