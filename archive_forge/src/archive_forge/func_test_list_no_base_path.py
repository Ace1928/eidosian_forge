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
def test_list_no_base_path(self):
    with mock.patch.object(self.Base, 'list') as list_mock:
        self.Base.find(self.cloud.compute, 'name')
        list_mock.assert_called_with(self.cloud.compute)