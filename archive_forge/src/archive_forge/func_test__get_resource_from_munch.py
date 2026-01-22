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
def test__get_resource_from_munch(self):
    cls = mock.Mock()
    res = mock.Mock(spec=resource.Resource)
    res._update = mock.Mock()
    cls._from_munch.return_value = res
    m = utils.Munch(answer=42)
    attrs = {'first': 'Brian', 'last': 'Curtin'}
    result = self.fake_proxy._get_resource(cls, m, **attrs)
    cls._from_munch.assert_called_once_with(m, connection=self.cloud)
    res._update.assert_called_once_with(**attrs)
    self.assertEqual(result, res)