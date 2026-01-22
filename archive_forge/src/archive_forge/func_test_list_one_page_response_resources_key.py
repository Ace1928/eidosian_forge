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
def test_list_one_page_response_resources_key(self):
    key = 'resources'

    class Test(self.test_class):
        resources_key = key
    id_value = 1
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {key: [{'id': id_value}]}
    mock_response.links = []
    self.session.get.return_value = mock_response
    sot = Test()
    results = list(sot.list(self.session))
    self.session.get.assert_called_once_with(self.base_path, headers={'Accept': 'application/json'}, params={}, microversion=None)
    self.assertEqual(1, len(results))
    self.assertEqual(id_value, results[0].id)
    self.assertIsInstance(results[0], self.test_class)