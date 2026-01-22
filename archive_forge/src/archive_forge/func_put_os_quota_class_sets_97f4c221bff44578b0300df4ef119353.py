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
def put_os_quota_class_sets_97f4c221bff44578b0300df4ef119353(self, body, **kw):
    assert list(body) == ['quota_class_set']
    return (200, {}, {'quota_class_set': {'metadata_items': 1, 'injected_file_content_bytes': 1, 'injected_file_path_bytes': 1, 'ram': 1, 'floating_ips': 1, 'instances': 1, 'injected_files': 1, 'cores': 1, 'key_pairs': 1, 'security_groups': 1, 'security_group_rules': 1}})