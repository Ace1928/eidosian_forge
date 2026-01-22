import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
def mock_request(self, path, verb, response=None, raw=False, status_code=200, req_headers=False, **kwargs):
    kwargs = dict(kwargs)
    if req_headers:
        if self.client is http.HTTPClient:
            kwargs['headers'] = {'X-Auth-Key': 'password', 'X-Auth-User': 'username'}
        else:
            kwargs['headers'] = {}
    reason = 'OK'
    if response:
        headers = {'content-type': 'application/json'}
        content = jsonutils.dumps(response)
    else:
        headers = {}
        content = None
    if status_code == 201:
        headers['location'] = 'http://heat.example.com/stacks/myStack'
    resp = fakes.FakeHTTPResponse(status_code, reason, headers, content)
    if self.client == http.SessionClient:
        request = self.SESSION
        self._results[request].append(resp)
        self._expect_call(request, path, verb, **kwargs)
    else:
        if raw:
            request = self.RAW
            self._results[request].append(resp)
        else:
            request = self.JSON
            self._results[request].append((resp, response))
        self._expect_call(request, verb, path, **kwargs)