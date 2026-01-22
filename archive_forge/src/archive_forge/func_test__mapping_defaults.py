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
def test__mapping_defaults(self):
    self.assertIn('location', resource.Resource._computed_mapping())
    self.assertIn('name', resource.Resource._body_mapping())
    self.assertIn('id', resource.Resource._body_mapping())