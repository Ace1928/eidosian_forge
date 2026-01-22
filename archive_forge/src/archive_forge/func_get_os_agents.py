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
def get_os_agents(self, **kw):
    hypervisor = kw.get('hypervisor', 'kvm')
    return (200, {}, {'agents': [{'hypervisor': hypervisor, 'os': 'win', 'architecture': 'x86', 'version': '7.0', 'url': 'xxx://xxxx/xxx/xxx', 'md5hash': 'add6bb58e139be103324d04d82d8f545', 'id': 1}, {'hypervisor': hypervisor, 'os': 'linux', 'architecture': 'x86', 'version': '16.0', 'url': 'xxx://xxxx/xxx/xxx1', 'md5hash': 'add6bb58e139be103324d04d82d8f546', 'id': 2}]})