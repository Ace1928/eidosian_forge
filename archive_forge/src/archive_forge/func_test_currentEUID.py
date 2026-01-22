import errno
import os.path
import shutil
import sys
import warnings
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.python import util
from twisted.python.filepath import FilePath
from twisted.test.test_process import MockOS
from twisted.trial.unittest import FailTest, TestCase
from twisted.trial.util import suppress as SUPPRESS
def test_currentEUID(self):
    """
        If the current euid is the same as the euid passed to L{util.switchUID},
        then initgroups does not get called, but a warning is issued.
        """
    euid = self.mockos.geteuid()
    util.switchUID(euid, None, True)
    self.assertEqual(self.initgroupsCalls, [])
    self.assertEqual(self.mockos.seteuidCalls, [])
    currentWarnings = self.flushWarnings([util.switchUID])
    self.assertEqual(len(currentWarnings), 1)
    self.assertIn('tried to drop privileges and seteuid %i' % euid, currentWarnings[0]['message'])
    self.assertIn('but euid is already %i' % euid, currentWarnings[0]['message'])