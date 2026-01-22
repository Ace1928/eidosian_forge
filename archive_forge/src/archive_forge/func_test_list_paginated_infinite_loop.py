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
def test_list_paginated_infinite_loop(self):
    q_limit = 1
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.links = {}
    mock_response.json.side_effect = [{'resources': [{'id': 1}]}, {'resources': [{'id': 1}]}]
    self.session.get.return_value = mock_response

    class Test(self.test_class):
        _query_mapping = resource.QueryParameters('limit')
    res = Test.list(self.session, paginated=True, limit=q_limit)
    self.assertRaises(exceptions.SDKException, list, res)