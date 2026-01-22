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
def test_service_cleanup_dry_run(self):
    self.assertTrue(self.sot._service_cleanup_del_res(self.delete_mock, self.res, dry_run=True))
    self.delete_mock.assert_not_called()