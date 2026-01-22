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
def test_bad_template_file(self):
    self.register_keystone_auth_fixture()
    failed_msg = 'Error parsing template '
    with tempfile.NamedTemporaryFile() as bad_json_file:
        bad_json_file.write(b'{foo:}')
        bad_json_file.flush()
        self.shell_error('stack-create ts -f %s' % bad_json_file.name, failed_msg, exception=exc.CommandError)
    with tempfile.NamedTemporaryFile() as bad_json_file:
        bad_json_file.write(b'{"foo": None}')
        bad_json_file.flush()
        self.shell_error('stack-create ts -f %s' % bad_json_file.name, failed_msg, exception=exc.CommandError)