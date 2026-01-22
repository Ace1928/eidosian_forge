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
def test_unknown_attrs_not_stored1(self):

    class Test(resource.Resource):
        _store_unknown_attrs_as_properties = True
    sot = Test.new(**{'dummy': 'value'})
    self.assertRaises(KeyError, sot.__getitem__, 'properties')