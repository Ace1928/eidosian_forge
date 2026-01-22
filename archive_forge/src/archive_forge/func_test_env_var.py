import os
import sys
from ..transport.http import ca_bundle
from . import TestCaseInTempDir, TestSkipped
def test_env_var(self):
    self.overrideEnv('CURL_CA_BUNDLE', 'foo.bar')
    self._make_file()
    self.assertEqual('foo.bar', ca_bundle.get_ca_path(use_cache=False))