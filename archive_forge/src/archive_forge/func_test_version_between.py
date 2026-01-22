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
def test_version_between(self):

    def good(minver, maxver, cand):
        self.assertTrue(discover.version_between(minver, maxver, cand))

    def bad(minver, maxver, cand):
        self.assertFalse(discover.version_between(minver, maxver, cand))

    def exc(excls, minver, maxver, cand):
        self.assertRaises(excls, discover.version_between, minver, maxver, cand)
    exc(ValueError, (1, 0), (1, 0), None)
    exc(ValueError, 'v1.0', '1.0', '')
    exc(TypeError, None, None, 'bogus')
    exc(TypeError, None, None, (1, 'two'))
    exc(TypeError, 'bogus', None, (1, 0))
    exc(TypeError, (1, 'two'), None, (1, 0))
    exc(TypeError, None, 'bogus', (1, 0))
    exc(TypeError, None, (1, 'two'), (1, 0))
    bad((2, 4), None, (1, 55))
    bad('v2.4', '', '2.3')
    bad('latest', None, (2, 3000))
    bad((2, discover.LATEST), '', 'v2.3000')
    bad((2, 3000), '', (1, discover.LATEST))
    bad('latest', None, 'v1000.latest')
    bad(None, (2, 4), (2, 5))
    bad('', 'v2.4', '2.5')
    bad(None, (2, discover.LATEST), (3, 0))
    bad('', '2000.latest', 'latest')
    good((1, 0), (1, 0), (1, 0))
    good('1.0', '2.9', '1.0')
    good('v1.0', 'v2.9', 'v2.9')
    good((1, 0), (1, 10), (1, 2))
    good('1', '2', '1.2')
    good(None, (2, 5), (2, 3))
    good('2.5', '', '2.6')
    good('', '', 'v1')
    good(None, None, (999, 999))
    good(None, None, 'latest')
    good((discover.LATEST, discover.LATEST), (discover.LATEST, discover.LATEST), (discover.LATEST, discover.LATEST))
    good((discover.LATEST, discover.LATEST), None, (discover.LATEST, discover.LATEST))
    good('', 'latest', 'latest')
    good('2.latest', '3.latest', '3.0')
    good('2.latest', None, (55, 66))
    good(None, '3.latest', '3.9999')