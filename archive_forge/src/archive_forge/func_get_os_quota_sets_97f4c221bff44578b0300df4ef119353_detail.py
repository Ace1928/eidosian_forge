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
def get_os_quota_sets_97f4c221bff44578b0300df4ef119353_detail(self, **kw):
    return (200, {}, {'quota_set': {'tenant_id': '97f4c221bff44578b0300df4ef119353', 'cores': {'in_use': 0, 'limit': 20, 'reserved': 0}, 'fixed_ips': {'in_use': 0, 'limit': -1, 'reserved': 0}, 'floating_ips': {'in_use': 0, 'limit': 10, 'reserved': 0}, 'injected_file_content_bytes': {'in_use': 0, 'limit': 10240, 'reserved': 0}, 'injected_file_path_bytes': {'in_use': 0, 'limit': 255, 'reserved': 0}, 'injected_files': {'in_use': 0, 'limit': 5, 'reserved': 0}, 'instances': {'in_use': 0, 'limit': 10, 'reserved': 0}, 'key_pairs': {'in_use': 0, 'limit': 100, 'reserved': 0}, 'metadata_items': {'in_use': 0, 'limit': 128, 'reserved': 0}, 'ram': {'in_use': 0, 'limit': 51200, 'reserved': 0}, 'security_group_rules': {'in_use': 0, 'limit': 20, 'reserved': 0}, 'security_groups': {'in_use': 0, 'limit': 10, 'reserved': 0}, 'server_group_members': {'in_use': 0, 'limit': 10, 'reserved': 0}, 'server_groups': {'in_use': 0, 'limit': 10, 'reserved': 0}}})