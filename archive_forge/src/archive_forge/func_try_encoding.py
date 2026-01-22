import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def try_encoding(self, encoding, fail=False):
    brz = self.run_bzr
    if fail:
        self.assertRaises(UnicodeEncodeError, self._mu.encode, encoding)
        encoded_msg = self._message.encode(encoding, 'replace')
    else:
        encoded_msg = self._message.encode(encoding)
    old_encoding = osutils._cached_user_encoding
    try:
        osutils._cached_user_encoding = 'ascii'
        out, err = brz('log', encoding=encoding)
        if not fail:
            self.assertNotEqual(-1, out.find(self._message))
        else:
            self.assertNotEqual(-1, out.find('Message with ?'))
    finally:
        osutils._cached_user_encoding = old_encoding