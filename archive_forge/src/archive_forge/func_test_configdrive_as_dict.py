from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_configdrive_as_dict(self):
    for target in ('rebuild', 'active'):
        self.session.put.reset_mock()
        result = self.node.set_provision_state(self.session, target, config_drive={'user_data': 'abcd'})
        self.assertIs(result, self.node)
        self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': target, 'configdrive': {'user_data': 'abcd'}}, headers=mock.ANY, microversion='1.56', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)