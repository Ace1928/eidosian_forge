import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
def test_file_only_contains_one_credential(self):
    credential1 = self.make_credential('consumer key')
    credential2 = self.make_credential('consumer key2')
    self.store.save(credential1, 'unique key 1')
    self.store.save(credential1, 'unique key 2')
    loaded = self.store.load('unique key 1')
    self.assertEqual(loaded.consumer.key, credential2.consumer.key)