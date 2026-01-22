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
def test__get_id_instance(self):

    class Test(resource.Resource):
        id = resource.Body('id')
    value = 'id'
    sot = Test(id=value)
    self.assertEqual(value, sot._get_id(sot))