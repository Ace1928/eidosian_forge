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
def get_servers_1234_os_instance_actions(self, **kw):
    action = {'instance_uuid': '1234', 'user_id': 'b968c25e04ab405f9fe4e6ca54cce9a5', 'start_time': '2013-03-25T13:45:09.000000', 'request_id': 'req-abcde12345', 'action': 'create', 'message': None, 'project_id': '04019601fe3648c0abd4f4abfb9e6106'}
    if self.api_version >= api_versions.APIVersion('2.58'):
        action['updated_at'] = '2013-03-25T13:50:09.000000'
    return (200, FAKE_RESPONSE_HEADERS, {'instanceActions': [action]})