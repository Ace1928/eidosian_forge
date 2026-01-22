import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
def test_unique_id_doesnt_matter(self):
    credential = self.make_credential('consumer key')
    self.store.save(credential, 'some key')
    credential2 = self.store.load('some other key')
    self.assertEqual(credential.consumer.key, credential2.consumer.key)