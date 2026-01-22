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
def test_user_agent(self):
    session = client_session.Session()
    self.stub_url('GET', text='response')
    resp = session.get(self.TEST_URL)
    self.assertTrue(resp.ok)
    self.assertRequestHeaderEqual('User-Agent', '%s %s' % ('run.py', client_session.DEFAULT_USER_AGENT))
    custom_agent = 'custom-agent/1.0'
    session = client_session.Session(user_agent=custom_agent)
    self.stub_url('GET', text='response')
    resp = session.get(self.TEST_URL)
    self.assertTrue(resp.ok)
    self.assertRequestHeaderEqual('User-Agent', '%s %s' % (custom_agent, client_session.DEFAULT_USER_AGENT))
    resp = session.get(self.TEST_URL, headers={'User-Agent': 'new-agent'})
    self.assertTrue(resp.ok)
    self.assertRequestHeaderEqual('User-Agent', 'new-agent')
    resp = session.get(self.TEST_URL, headers={'User-Agent': 'new-agent'}, user_agent='overrides-agent')
    self.assertTrue(resp.ok)
    self.assertRequestHeaderEqual('User-Agent', 'overrides-agent')
    with mock.patch.object(sys, 'argv', []):
        session = client_session.Session()
        resp = session.get(self.TEST_URL)
        self.assertTrue(resp.ok)
        self.assertRequestHeaderEqual('User-Agent', client_session.DEFAULT_USER_AGENT)
    with mock.patch.object(sys, 'argv', ['']):
        session = client_session.Session()
        resp = session.get(self.TEST_URL)
        self.assertTrue(resp.ok)
        self.assertRequestHeaderEqual('User-Agent', client_session.DEFAULT_USER_AGENT)