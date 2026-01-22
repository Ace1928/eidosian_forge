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
def test_transpose_not_in_query(self):
    location = 'location'
    mapping = {'first_name': 'first-name', 'pet_name': {'name': 'pet'}, 'answer': {'name': 'answer', 'type': int}}
    sot = resource.QueryParameters(location, **mapping)
    result = sot._transpose({'location': 'Brooklyn'}, mock.sentinel.resource_type)
    self.assertEqual({'location': 'Brooklyn'}, result)