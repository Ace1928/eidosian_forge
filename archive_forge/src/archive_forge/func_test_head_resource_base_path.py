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
def test_head_resource_base_path(self):
    base_path = 'dummy'
    rv = self.sot._head(HeadableResource, self.res, base_path=base_path)
    self.res.head.assert_called_with(self.sot, base_path=base_path)
    self.assertEqual(rv, self.fake_result)