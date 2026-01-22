import copy
import functools
import random
import http.client
from oslo_serialization import jsonutils
from testtools import matchers as tt_matchers
import webob
from keystone.api import discovery
from keystone.common import json_home
from keystone.tests import unit
def test_json_home_v3(self):
    exp_json_home_data = {'resources': V3_JSON_HOME_RESOURCES}
    self._test_json_home('/v3', exp_json_home_data)