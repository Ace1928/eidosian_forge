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
def test_head_with_microversion(self):

    class Test(resource.Resource):
        service = self.service_name
        base_path = self.base_path
        allow_head = True
        _max_microversion = '1.42'
    sot = Test(id='id')
    sot._prepare_request = mock.Mock(return_value=self.request)
    sot._translate_response = mock.Mock()
    result = sot.head(self.session)
    sot._prepare_request.assert_called_once_with(base_path=None)
    self.session.head.assert_called_once_with(self.request.url, microversion='1.42')
    self.assertEqual(sot.microversion, '1.42')
    sot._translate_response.assert_called_once_with(self.response, has_body=False)
    self.assertEqual(result, sot)