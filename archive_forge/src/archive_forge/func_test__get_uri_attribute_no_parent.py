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
def test__get_uri_attribute_no_parent(self):

    class Child(resource.Resource):
        something = resource.Body('something')
    attr = 'something'
    value = 'nothing'
    child = Child(something=value)
    result = self.fake_proxy._get_uri_attribute(child, None, attr)
    self.assertEqual(value, result)