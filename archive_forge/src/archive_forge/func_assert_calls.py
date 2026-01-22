import collections
import os
import tempfile
import time
import urllib
import uuid
import fixtures
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from requests import structures
from requests_mock.contrib import fixture as rm_fixture
import openstack.cloud
import openstack.config as occ
import openstack.connection
from openstack.fixture import connection as os_fixture
from openstack.tests import base
from openstack.tests import fakes
def assert_calls(self, stop_after=None, do_count=True):
    for x, (call, history) in enumerate(zip(self.calls, self.adapter.request_history)):
        if stop_after and x > stop_after:
            break
        call_uri_parts = urllib.parse.urlparse(call['url'])
        history_uri_parts = urllib.parse.urlparse(history.url)
        self.assertEqual((call['method'], call_uri_parts.scheme, call_uri_parts.netloc, call_uri_parts.path, call_uri_parts.params, urllib.parse.parse_qs(call_uri_parts.query)), (history.method, history_uri_parts.scheme, history_uri_parts.netloc, history_uri_parts.path, history_uri_parts.params, urllib.parse.parse_qs(history_uri_parts.query)), 'REST mismatch on call %(index)d. Expected %(call)r. Got %(history)r). NOTE: query string order differences wont cause mismatch' % {'index': x, 'call': '{method} {url}'.format(method=call['method'], url=call['url']), 'history': '{method} {url}'.format(method=history.method, url=history.url)})
        if 'json' in call:
            self.assertEqual(call['json'], history.json(), 'json content mismatch in call {index}'.format(index=x))
        if 'headers' in call:
            for key, value in call['headers'].items():
                self.assertEqual(value, history.headers[key], 'header mismatch in call {index}'.format(index=x))
    if do_count:
        self.assertEqual(len(self.calls), len(self.adapter.request_history), "Expected:\n{}'\nGot:\n{}".format('\n'.join([c['url'] for c in self.calls]), '\n'.join([h.url for h in self.adapter.request_history])))