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
def test_unknown_attrs_under_props_set(self):

    class Test(resource.Resource):
        properties = resource.Body('properties')
        _store_unknown_attrs_as_properties = True
    sot = Test.new(**{'dummy': 'value'})
    sot['properties'] = {'dummy': 'new_value'}
    self.assertEqual('new_value', sot['properties']['dummy'])
    sot.properties = {'dummy': 'new_value1'}
    self.assertEqual('new_value1', sot['properties']['dummy'])