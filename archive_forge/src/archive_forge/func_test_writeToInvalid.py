import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
def test_writeToInvalid(self):
    """
        Verify that writing with L{fdesc.writeToFD} when the write end is
        closed results in a connection lost indicator.
        """
    os.close(self.w)
    self.assertEqual(self.write(b's'), fdesc.CONNECTION_LOST)