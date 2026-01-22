from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_validate_no_failure(self):
    self.session.get.return_value.json.return_value = {'boot': {'result': False}, 'console': {'result': False, 'reason': 'Not configured'}, 'deploy': {'result': False, 'reason': 'No deploy for you'}, 'inspect': {'result': None, 'reason': 'Not supported'}, 'power': {'result': True}}
    result = self.node.validate(self.session, required=None)
    self.assertTrue(result['power'].result)
    self.assertFalse(result['power'].reason)
    for iface in ('deploy', 'console', 'inspect'):
        self.assertIsNot(True, result[iface].result)
        self.assertTrue(result[iface].reason)
    self.assertFalse(result['boot'].result)
    self.assertIsNone(result['boot'].reason)