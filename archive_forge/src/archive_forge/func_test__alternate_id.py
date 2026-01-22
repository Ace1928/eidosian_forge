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
def test__alternate_id(self):

    class Test(resource.Resource):
        alt = resource.Body('the_alt', alternate_id=True)
    self.assertEqual('the_alt', Test._alternate_id())
    value1 = 'lol'
    sot = Test(alt=value1)
    self.assertEqual(sot.alt, value1)
    self.assertEqual(sot.id, value1)
    value2 = 'rofl'
    sot = Test(the_alt=value2)
    self.assertEqual(sot.alt, value2)
    self.assertEqual(sot.id, value2)