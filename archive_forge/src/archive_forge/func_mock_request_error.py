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
def mock_request_error(self, path, verb, error):
    raw = verb == 'DELETE'
    if self.client == http.SessionClient:
        request = self.SESSION
        self._expect_call(request, path, verb)
    else:
        if raw:
            request = self.RAW
        else:
            request = self.JSON
        self._expect_call(request, verb, path)
    self._results[request].append(error)