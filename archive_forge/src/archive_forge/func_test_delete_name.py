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
def test_delete_name(self):
    name = 'name'
    expected_value = '123'

    class Parent:
        _example = {name: expected_value}
    instance = Parent()
    sot = TestComponent.ExampleComponent('name')
    sot.__delete__(instance)
    self.assertNotIn(name, instance._example)