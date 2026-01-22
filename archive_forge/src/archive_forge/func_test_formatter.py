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
def test_formatter(self):
    """
        If C{showAttributes} has an item that is a 2-tuple, C{__str__} renders
        the first item in the tuple as a key and the result of calling the
        second item with the value of the attribute named by the first item as
        the value.
        """

    class Foo(util.FancyStrMixin):
        showAttributes = ('first', ('second', lambda value: repr(value[::-1])))
        first = 'hello'
        second = 'world'
    self.assertEqual("<Foo first='hello' second='dlrow'>", str(Foo()))