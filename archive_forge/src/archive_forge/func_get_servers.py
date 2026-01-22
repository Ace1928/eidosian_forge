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
def get_servers(self, **kw):
    servers = {'servers': [{'id': '1234', 'name': 'sample-server'}, {'id': '5678', 'name': 'sample-server2'}, {'id': '9014', 'name': 'help'}]}
    if self.api_version >= api_versions.APIVersion('2.69'):
        servers['servers'].append({'id': '9015', 'status': 'UNKNOWN', 'links': [{'href': 'http://fake/v2.1/', 'rel': 'self'}, {'href': 'http://fake', 'rel': 'bookmark'}]})
    return (200, {}, servers)