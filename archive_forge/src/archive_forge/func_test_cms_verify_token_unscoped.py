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
def test_cms_verify_token_unscoped(self):
    cms_content = cms.token_to_cms(self.examples.SIGNED_TOKEN_UNSCOPED)
    self.assertTrue(cms.cms_verify(cms_content, self.examples.SIGNING_CERT_FILE, self.examples.SIGNING_CA_FILE))