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
def test_parsable_malformed_error(self):
    self.register_keystone_auth_fixture()
    invalid_json = 'ERROR: {Invalid JSON Error.'
    self.mock_request_error('/stacks/bad', 'GET', exc.HTTPBadRequest(invalid_json))
    e = self.assertRaises(exc.HTTPException, self.shell, 'stack-show bad')
    self.assertEqual('ERROR: ' + invalid_json, str(e))