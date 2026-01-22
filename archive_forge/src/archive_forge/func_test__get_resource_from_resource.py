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
def test__get_resource_from_resource(self):
    res = mock.Mock(spec=resource.Resource)
    res._update = mock.Mock()
    attrs = {'first': 'Brian', 'last': 'Curtin'}
    result = self.fake_proxy._get_resource(resource.Resource, res, **attrs)
    res._update.assert_called_once_with(**attrs)
    self.assertEqual(result, res)