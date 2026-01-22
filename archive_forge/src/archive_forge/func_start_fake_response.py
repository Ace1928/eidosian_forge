from unittest import mock
import urllib.parse
import fixtures
from oslo_serialization import jsonutils
import requests
from requests_mock.contrib import fixture as rm_fixture
from testtools import matchers
import webob
from keystonemiddleware import s3_token
from keystonemiddleware.tests.unit import utils
def start_fake_response(self, status, headers):
    self.response_status = int(status.split(' ', 1)[0])
    self.response_headers = dict(headers)