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
def post_servers_1234_os_volume_attachments(self, **kw):
    attachment = {'device': '/dev/vdb', 'volumeId': 2}
    if self.api_version >= api_versions.APIVersion('2.70'):
        attachment['tag'] = 'test-tag'
    if self.api_version >= api_versions.APIVersion('2.79'):
        attachment['delete_on_termination'] = True
    return (200, FAKE_RESPONSE_HEADERS, {'volumeAttachment': attachment})