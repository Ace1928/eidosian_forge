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
def test__getattribute__id_in_body(self):
    id = 'lol'
    sot = resource.Resource(id=id)
    result = getattr(sot, 'id')
    self.assertEqual(result, id)