import os
from unittest import mock
import fixtures
from oslo_serialization import jsonutils
import requests
from requests_mock.contrib import fixture as requests_mock_fixture
import testscenarios
import testtools
def assert_request_id(self, request_id_mixin, request_id_list):
    self.assertEqual(request_id_list, request_id_mixin.request_ids)