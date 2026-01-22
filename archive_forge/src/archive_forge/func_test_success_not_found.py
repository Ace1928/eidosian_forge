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
def test_success_not_found(self):
    response = mock.Mock()
    response.headers = {}
    response.status_code = 404
    res = mock.Mock()
    res.fetch.side_effect = [res, res, exceptions.ResourceNotFound('Not Found', response)]
    result = resource.wait_for_delete(self.cloud.compute, res, 1, 3)
    self.assertEqual(result, res)