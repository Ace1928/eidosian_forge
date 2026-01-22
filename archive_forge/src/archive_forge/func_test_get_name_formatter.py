import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_get_name_formatter(self):
    name = 'name'
    value = '123'
    expected_result = 'one hundred twenty three'

    class Parent:
        _example = {name: value}

    class FakeFormatter(format.Formatter):

        @classmethod
        def deserialize(cls, value):
            return expected_result
    instance = Parent()
    sot = TestComponent.ExampleComponent('name', type=FakeFormatter)
    result = sot.__get__(instance, None)
    self.assertEqual(expected_result, result)