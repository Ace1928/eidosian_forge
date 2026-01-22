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
def test_delete_ignore_missing(self):
    self.res.delete.side_effect = exceptions.ResourceNotFound(message='test', http_status=404)
    rv = self.sot._delete(DeleteableResource, self.fake_id)
    self.assertIsNone(rv)