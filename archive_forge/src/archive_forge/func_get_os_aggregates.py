import copy
import datetime
import re
from unittest import mock
from urllib import parse
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client as base_client
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils
from novaclient.v2 import client
def get_os_aggregates(self, *kw):
    response = (200, {}, {'aggregates': [{'id': '1', 'name': 'test', 'availability_zone': 'nova1'}, {'id': '2', 'name': 'test2', 'availability_zone': 'nova1'}, {'id': '3', 'name': 'test3', 'metadata': {'test': 'dup', 'none_key': 'Nine'}}]})
    if self.api_version >= api_versions.APIVersion('2.41'):
        aggregates = response[2]['aggregates']
        aggregates[0]['uuid'] = '80785864-087b-45a5-a433-b20eac9b58aa'
        aggregates[1]['uuid'] = '30827713-5957-4b68-8fd3-ccaddb568c24'
        aggregates[2]['uuid'] = '9a651b22-ce3f-4a87-acd7-98446ef591c4'
    return response