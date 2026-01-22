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
def test_failed_auth(self):
    self.register_keystone_auth_fixture()
    failed_msg = 'Unable to authenticate user with credentials provided'
    with mock.patch.object(http.SessionClient, 'request', side_effect=exc.Unauthorized(failed_msg)) as sc:
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)
        self.shell_error('stack-list', failed_msg, exception=exc.Unauthorized)
        sc.assert_called_once_with('/stacks?', 'GET')