import errno
import os
import subprocess
from unittest import mock
import testresources
from testtools import matchers
from keystoneclient.common import cms
from keystoneclient import exceptions
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
def test_asn1_token(self):
    self.assertTrue(cms.is_asn1_token(self.examples.SIGNED_TOKEN_SCOPED))
    self.assertFalse(cms.is_asn1_token('FOOBAR'))