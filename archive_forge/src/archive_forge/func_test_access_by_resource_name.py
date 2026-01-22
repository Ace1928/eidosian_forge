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
def test_access_by_resource_name(self):

    class Test(resource.Resource):
        blah = resource.Body('blah_resource')
    sot = Test(blah='dummy')
    result = sot['blah_resource']
    self.assertEqual(result, sot.blah)