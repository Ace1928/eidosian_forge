import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_setting_variables_on_request(self):
    response = uuid.uuid4().hex
    self.stub_url('GET', text=response)
    adpt = self._create_loaded_adapter()
    resp = adpt.get('/')
    self.assertEqual(resp.text, response)
    self._verify_endpoint_called(adpt)
    self.assertEqual(self.ALLOW, adpt.auth.endpoint_arguments['allow'])
    self.assertTrue(adpt.auth.get_token_called)
    self.assertRequestHeaderEqual('User-Agent', self.USER_AGENT)