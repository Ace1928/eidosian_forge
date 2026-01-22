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
def test_create_synced(self):
    attrs = {'hey': 1, 'hi': 2, 'hello': 3}
    sync = True
    sot = resource._ComponentManager(attributes=attrs, synchronized=sync)
    self.assertEqual(attrs, sot.attributes)
    self.assertEqual(set(), sot._dirty)