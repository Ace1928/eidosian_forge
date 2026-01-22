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
def get_servers_1234_os_interface(self, **kw):
    attachments = {'interfaceAttachments': [{'port_state': 'ACTIVE', 'net_id': 'net-id-1', 'port_id': 'port-id-1', 'mac_address': 'aa:bb:cc:dd:ee:ff', 'fixed_ips': [{'ip_address': '1.2.3.4'}]}, {'port_state': 'ACTIVE', 'net_id': 'net-id-1', 'port_id': 'port-id-1', 'mac_address': 'aa:bb:cc:dd:ee:ff', 'fixed_ips': [{'ip_address': '1.2.3.4'}]}]}
    if self.api_version >= api_versions.APIVersion('2.70'):
        for attachment in attachments['interfaceAttachments']:
            attachment['tag'] = 'test-tag'
    return (200, {}, attachments)