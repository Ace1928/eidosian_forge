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
def test_put_create_with_microversion(self):

    class Test(resource.Resource):
        service = self.service_name
        base_path = self.base_path
        allow_create = True
        create_method = 'PUT'
        _max_microversion = '1.42'
    self._test_create(Test, requires_id=True, prepend_key=True, microversion='1.42')