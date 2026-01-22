from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import _proxy
from openstack.baremetal_introspection.v1 import introspection
from openstack.baremetal_introspection.v1 import introspection_rule
from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit import test_proxy_base
def test_create_introspection(self, mock_create):
    self.proxy.start_introspection('abcd')
    mock_create.assert_called_once_with(mock.ANY, self.proxy)
    introspect = mock_create.call_args[0][0]
    self.assertEqual('abcd', introspect.id)