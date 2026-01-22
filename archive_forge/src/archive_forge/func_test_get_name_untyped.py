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
def test_get_name_untyped(self):
    name = 'name'
    expected_result = 123

    class Parent:
        _example = {name: expected_result}
    instance = Parent()
    sot = TestComponent.ExampleComponent('name')
    result = sot.__get__(instance, None)
    self.assertEqual(expected_result, result)