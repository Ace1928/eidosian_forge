from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_address_scope_handle_delete_not_found(self):
    addrs_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    self.my_address_scope.resource_id = addrs_id
    not_found = self.neutronclient.NotFound
    self.neutronclient.delete_address_scope.side_effect = not_found
    self.assertIsNone(self.my_address_scope.handle_delete())
    self.neutronclient.delete_address_scope.assert_called_once_with(self.my_address_scope.resource_id)