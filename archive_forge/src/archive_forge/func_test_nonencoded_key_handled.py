import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
def test_nonencoded_key_handled(self):

    class UnencodedInMemoryKeyring(InMemoryKeyring):

        def get_password(self, service, username):
            pw = super(UnencodedInMemoryKeyring, self).get_password(service, username)
            return b64decode(pw[5:])
    self.keyring = UnencodedInMemoryKeyring()
    with fake_keyring(self.keyring):
        credential = self.make_credential('consumer key')
        self.store.save(credential, 'unique key')
        credential2 = self.store.load('unique key')
        self.assertEqual(credential.consumer.key, credential2.consumer.key)
        self.assertEqual(credential.consumer.secret, credential2.consumer.secret)