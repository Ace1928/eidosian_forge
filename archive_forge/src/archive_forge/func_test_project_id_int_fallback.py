import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_project_id_int_fallback(self):
    bad_url = 'https://compute.example.com/v2/123456'
    epd = discover.EndpointData(catalog_url=bad_url)
    self.assertEqual((2, 0), epd.api_version)