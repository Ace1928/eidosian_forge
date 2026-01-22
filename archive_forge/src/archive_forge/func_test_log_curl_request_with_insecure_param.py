import functools
import json
import logging
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import session
from keystoneauth1 import token_endpoint
from oslo_utils import encodeutils
import requests
from requests_mock.contrib import fixture
from urllib import parse
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
import testtools
from testtools import matchers
import types
import glanceclient
from glanceclient.common import http
from glanceclient.tests import utils
@mock.patch('glanceclient.common.http.LOG.debug')
def test_log_curl_request_with_insecure_param(self, mock_log):
    headers = {'header1': 'value1'}
    http_client_object = http.HTTPClient(self.ssl_endpoint, insecure=True, token='fake-token')
    http_client_object.log_curl_request('GET', '/api/v1/', headers, None, None)
    self.assertTrue(mock_log.called, 'LOG.debug never called')
    self.assertTrue(mock_log.call_args[0], 'LOG.debug called with no arguments')
    self.assertThat(mock_log.call_args[0][0], matchers.MatchesRegex('.*\\s-k\\s.*'), 'no -k option in curl command')