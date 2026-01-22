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
def test_list_multi_page_response_not_paginated(self):
    ids = [1, 2]
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = [{'resources': [{'id': ids[0]}]}, {'resources': [{'id': ids[1]}]}]
    self.session.get.return_value = mock_response
    results = list(self.sot.list(self.session, paginated=False))
    self.assertEqual(1, len(results))
    self.assertEqual(ids[0], results[0].id)
    self.assertIsInstance(results[0], self.test_class)