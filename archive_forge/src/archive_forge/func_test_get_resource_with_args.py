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
def test_get_resource_with_args(self):
    args = {'key': 'value'}
    rv = self.sot._get(RetrieveableResource, self.res, **args)
    self.res._update.assert_called_once_with(**args)
    self.res.fetch.assert_called_with(self.sot, requires_id=True, base_path=None, skip_cache=mock.ANY, error_message=mock.ANY)
    self.assertEqual(rv, self.fake_result)