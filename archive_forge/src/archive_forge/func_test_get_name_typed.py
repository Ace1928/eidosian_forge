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
def test_get_name_typed(self):
    name = 'name'
    value = '123'

    class Parent:
        _example = {name: value}
    instance = Parent()
    sot = TestComponent.ExampleComponent('name', type=int)
    result = sot.__get__(instance, None)
    self.assertEqual(int(value), result)