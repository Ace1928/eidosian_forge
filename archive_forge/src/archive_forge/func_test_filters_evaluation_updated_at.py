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
def test_filters_evaluation_updated_at(self):
    self.assertTrue(self.sot._service_cleanup_resource_filters_evaluation(self.res, filters={'updated_at': '2020-02-03T00:00:00'}))