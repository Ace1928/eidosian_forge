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
def test_setitem_new(self):
    key = 'key'
    value = 'value'
    sot = resource._ComponentManager()
    sot.__setitem__(key, value)
    self.assertIn(key, sot.attributes)
    self.assertIn(key, sot.dirty)