import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_unresumable_write_group(self):
    repo = 'dummy repo'
    wg_tokens = ['token']
    reason = 'a reason'
    err = errors.UnresumableWriteGroup(repo, wg_tokens, reason)
    self.assertEqual("Repository dummy repo cannot resume write group ['token']: a reason", str(err))