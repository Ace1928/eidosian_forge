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
def test_cms_hash_token_sha256(self):
    """Can also hash with sha256."""
    token = self.examples.SIGNED_TOKEN_SCOPED
    token_id = cms.cms_hash_token(token, mode='sha256')
    self.assertThat(token_id, matchers.HasLength(64))