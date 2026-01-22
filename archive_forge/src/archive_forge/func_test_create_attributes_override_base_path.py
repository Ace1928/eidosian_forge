import copy
import queue
from unittest import mock
from keystoneauth1 import session
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_create_attributes_override_base_path(self):
    CreateableResource.new = mock.Mock(return_value=self.res)
    base_path = 'dummy'
    attrs = {'x': 1, 'y': 2, 'z': 3}
    rv = self.sot._create(CreateableResource, base_path=base_path, **attrs)
    self.assertEqual(rv, self.fake_result)
    CreateableResource.new.assert_called_once_with(connection=self.cloud, **attrs)
    self.res.create.assert_called_once_with(self.sot, base_path=base_path)