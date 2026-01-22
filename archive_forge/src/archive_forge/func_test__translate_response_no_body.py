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
def test__translate_response_no_body(self):

    class Test(resource.Resource):
        attr = resource.Header('attr')
    response = FakeResponse({}, headers={'attr': 'value'})
    sot = Test()
    sot._translate_response(response, has_body=False)
    self.assertEqual(dict(), sot._header.dirty)
    self.assertEqual('value', sot.attr)