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
def test_runWithWarningsSuppressedUnfiltered(self):
    """
        Warnings from the function called by C{runWithWarningsSuppressed} are
        not suppressed if they do not match the passed in filter.
        """
    filters = [(('ignore', '.*foo.*'), {}), (('ignore', '.*bar.*'), {})]
    self.runWithWarningsSuppressed(filters, warnings.warn, "don't ignore")
    self.assertEqual(["don't ignore"], [w['message'] for w in self.flushWarnings()])