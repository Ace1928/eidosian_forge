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
def test_cms_hash_token_default_md5(self):
    """The default hash method is md5."""
    token = self.examples.SIGNED_TOKEN_SCOPED
    token_id_default = cms.cms_hash_token(token)
    token_id_md5 = cms.cms_hash_token(token, mode='md5')
    self.assertThat(token_id_default, matchers.Equals(token_id_md5))
    self.assertThat(token_id_default, matchers.HasLength(32))