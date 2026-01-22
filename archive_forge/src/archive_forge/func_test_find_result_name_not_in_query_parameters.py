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
def test_find_result_name_not_in_query_parameters(self):
    with mock.patch.object(self.one_result, 'existing', side_effect=self.OneResult.existing) as mock_existing, mock.patch.object(self.one_result, 'list', side_effect=self.OneResult.list) as mock_list:
        self.assertEqual(self.result, self.one_result.find(self.cloud.compute, 'name'))
        mock_existing.assert_called_once_with(id='name', connection=mock.ANY)
        mock_list.assert_called_once_with(mock.ANY)