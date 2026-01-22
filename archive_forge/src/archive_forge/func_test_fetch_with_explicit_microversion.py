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
def test_fetch_with_explicit_microversion(self):

    class Test(resource.Resource):
        service = self.service_name
        base_path = self.base_path
        allow_fetch = True
        _max_microversion = '1.99'
    sot = Test(id='id')
    sot._prepare_request = mock.Mock(return_value=self.request)
    sot._translate_response = mock.Mock()
    result = sot.fetch(self.session, microversion='1.42')
    sot._prepare_request.assert_called_once_with(requires_id=True, base_path=None)
    self.session.get.assert_called_once_with(self.request.url, microversion='1.42', params={}, skip_cache=False)
    self.assertEqual(sot.microversion, '1.42')
    sot._translate_response.assert_called_once_with(self.response)
    self.assertEqual(result, sot)