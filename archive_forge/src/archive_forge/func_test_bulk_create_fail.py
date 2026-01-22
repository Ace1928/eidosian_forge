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
def test_bulk_create_fail(self):

    class Test(resource.Resource):
        service = self.service_name
        base_path = self.base_path
        create_method = 'POST'
        allow_create = False
        resources_key = 'tests'
    self.assertRaises(exceptions.MethodNotSupported, Test.bulk_create, self.session, [{'name': 'name'}])