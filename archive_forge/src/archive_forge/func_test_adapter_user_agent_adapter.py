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
def test_adapter_user_agent_adapter(self):
    sess = client_session.Session()
    adap = adapter.Adapter(client_name='testclient', client_version='4.5.6', session=sess)
    url = 'http://keystone.test.com'
    self.requests_mock.get(url)
    adap.get(url)
    agent = 'testclient/4.5.6'
    self.assertEqual(agent + ' ' + client_session.DEFAULT_USER_AGENT, self.requests_mock.last_request.headers['User-Agent'])