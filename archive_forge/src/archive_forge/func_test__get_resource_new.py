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
def test__get_resource_new(self):
    value = 'hello'
    fake_type = mock.Mock(spec=resource.Resource)
    fake_type.new = mock.Mock(return_value=value)
    attrs = {'first': 'Brian', 'last': 'Curtin'}
    result = self.fake_proxy._get_resource(fake_type, None, **attrs)
    fake_type.new.assert_called_with(connection=self.cloud, **attrs)
    self.assertEqual(value, result)