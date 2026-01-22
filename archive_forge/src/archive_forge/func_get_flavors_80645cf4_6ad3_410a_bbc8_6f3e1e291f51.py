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
def get_flavors_80645cf4_6ad3_410a_bbc8_6f3e1e291f51(self, **kw):
    raise exceptions.NotFound('404')