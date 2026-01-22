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
def test_single_match_by_id(self):
    the_id = 'Brian'
    match = mock.Mock(spec=resource.Resource)
    match.id = the_id
    result = resource.Resource._get_one_match(the_id, [match])
    self.assertIs(result, match)