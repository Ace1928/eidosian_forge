from unittest import mock
from urllib import parse
import ddt
import fixtures
from requests_mock.contrib import fixture as requests_mock_fixture
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import client
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient import utils as cinderclient_utils
from cinderclient.v3 import attachments
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
def test_default_type_list_project(self):
    self.run_command('--os-volume-api-version 3.62 default-type-list --project-id 629632e7-99d2-4c40-9ae3-106fa3b1c9b7')
    self.assert_called('GET', 'v3/default-types/629632e7-99d2-4c40-9ae3-106fa3b1c9b7')