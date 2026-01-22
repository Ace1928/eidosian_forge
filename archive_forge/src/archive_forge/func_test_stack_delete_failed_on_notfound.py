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
def test_stack_delete_failed_on_notfound(self):
    self.register_keystone_auth_fixture()
    self.mock_request_error('/stacks/teststack1/1', 'DELETE', exc.HTTPNotFound())
    error = self.assertRaises(exc.CommandError, self.shell, 'stack-delete teststack1/1')
    self.assertIn('Unable to delete 1 of the 1 stacks.', str(error))