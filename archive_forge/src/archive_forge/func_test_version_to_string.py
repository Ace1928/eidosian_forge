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
def test_version_to_string(self):

    def assert_string(out, inp):
        self.assertEqual(out, discover.version_to_string(inp))
    assert_string('latest', (discover.LATEST,))
    assert_string('latest', (discover.LATEST, discover.LATEST))
    assert_string('latest', (discover.LATEST, discover.LATEST, discover.LATEST))
    assert_string('1', (1,))
    assert_string('1.2', (1, 2))
    assert_string('1.latest', (1, discover.LATEST))