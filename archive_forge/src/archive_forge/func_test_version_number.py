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
def test_version_number(self):

    def assertVersion(out, inp):
        self.assertEqual(out, discover.normalize_version_number(inp))

    def versionRaises(inp):
        self.assertRaises(TypeError, discover.normalize_version_number, inp)
    assertVersion((1, 2), 'v1.2')
    assertVersion((11, 0), 'v11')
    assertVersion((1, 2), '1.2')
    assertVersion((1, 5, 1), '1.5.1')
    assertVersion((1, 0), '1')
    assertVersion((1, 0), 1)
    assertVersion((5, 2), 5.2)
    assertVersion((3, 20), '3.20')
    assertVersion((6, 1), (6, 1))
    assertVersion((1, 40), [1, 40])
    assertVersion((1, 0), (1,))
    assertVersion((1, 0), ['1'])
    assertVersion((discover.LATEST, discover.LATEST), 'latest')
    assertVersion((discover.LATEST, discover.LATEST), ['latest'])
    assertVersion((discover.LATEST, discover.LATEST), discover.LATEST)
    assertVersion((discover.LATEST, discover.LATEST), (discover.LATEST,))
    assertVersion((10, discover.LATEST), '10.latest')
    assertVersion((10, discover.LATEST), (10, 'latest'))
    assertVersion((10, discover.LATEST), (10, discover.LATEST))
    versionRaises(None)
    versionRaises('hello')
    versionRaises('1.a')
    versionRaises('vacuum')
    versionRaises('')
    versionRaises(('1', 'a'))