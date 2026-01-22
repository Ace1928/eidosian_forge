import io
import re
import sys
from unittest import mock
import fixtures
from keystoneauth1 import fixture
from testtools import matchers
from zunclient import api_versions
from zunclient import exceptions
import zunclient.shell
from zunclient.tests.unit import utils
@mock.patch('zunclient.client.Client')
def test_container_format_env(self, mock_client):
    self.make_env()
    self.shell('create --environment key=value test')
    _, create_args = mock_client.return_value.containers.create.call_args
    self.assertEqual({'key': 'value'}, create_args['environment'])