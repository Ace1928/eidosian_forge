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
def put_os_agents_1(self, body, **kw):
    return (200, {}, {'agent': {'url': '/yyy/yyyy/yyyy', 'version': '8.0', 'md5hash': 'add6bb58e139be103324d04d82d8f546', 'id': 1}})