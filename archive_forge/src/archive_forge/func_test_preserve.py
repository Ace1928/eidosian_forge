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
def test_preserve(self):
    """
        L{util.InsensitiveDict} preserves the case of keys if constructed with
        C{preserve=True}.
        """
    dct = util.InsensitiveDict({'Foo': 'bar', 1: 2, 'fnz': {1: 2}}, preserve=1)
    self.assertEqual(dct['fnz'], {1: 2})
    self.assertEqual(dct['foo'], 'bar')
    self.assertEqual(dct.copy(), dct)
    self.assertEqual(dct['foo'], dct.get('Foo'))
    self.assertIn(1, dct)
    self.assertIn('foo', dct)
    result = eval(repr(dct), {'dct': dct, 'InsensitiveDict': util.InsensitiveDict})
    self.assertEqual(result, dct)
    keys = ['Foo', 'fnz', 1]
    for x in keys:
        self.assertIn(x, dct.keys())
        self.assertIn((x, dct[x]), dct.items())
    self.assertEqual(len(keys), len(dct))
    del dct[1]
    del dct['foo']
    self.assertEqual(dct.keys(), ['fnz'])