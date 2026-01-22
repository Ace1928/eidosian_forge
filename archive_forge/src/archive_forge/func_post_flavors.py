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
def post_flavors(self, body, **kw):
    return (202, FAKE_RESPONSE_HEADERS, {'flavor': self.get_flavors_detail(is_public='None')[2]['flavors'][0]})