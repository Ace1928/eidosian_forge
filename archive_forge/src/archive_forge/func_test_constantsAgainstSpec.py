import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
def test_constantsAgainstSpec(self):
    """
        The constants used by the SFTP protocol implementation match those
        found by searching through the spec.
        """
    constants = {}
    for excerpt in self.filexferSpecExcerpts:
        for line in excerpt.splitlines():
            m = re.match('^\\s*#define SSH_([A-Z_]+)\\s+([0-9x]*)\\s*$', line)
            if m:
                constants[m.group(1)] = int(m.group(2), 0)
    self.assertTrue(len(constants) > 0, 'No constants found (the test must be buggy).')
    for k, v in constants.items():
        self.assertEqual(v, getattr(filetransfer, k))