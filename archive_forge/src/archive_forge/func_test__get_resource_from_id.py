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
def test__get_resource_from_id(self):
    id = 'eye dee'
    value = 'hello'
    attrs = {'first': 'Brian', 'last': 'Curtin'}

    class Fake:
        call = {}

        @classmethod
        def new(cls, **kwargs):
            cls.call = kwargs
            return value
    result = self.fake_proxy._get_resource(Fake, id, **attrs)
    self.assertDictEqual(dict(id=id, connection=mock.ANY, **attrs), Fake.call)
    self.assertEqual(value, result)