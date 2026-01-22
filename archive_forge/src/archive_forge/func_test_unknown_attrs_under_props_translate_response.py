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
def test_unknown_attrs_under_props_translate_response(self):

    class Test(resource.Resource):
        properties = resource.Body('properties')
        _store_unknown_attrs_as_properties = True
    body = {'dummy': 'value', 'properties': 'a,b,c'}
    response = FakeResponse(body)
    sot = Test()
    sot._translate_response(response, has_body=True)
    self.assertDictEqual({'dummy': 'value', 'properties': 'a,b,c'}, sot.properties)