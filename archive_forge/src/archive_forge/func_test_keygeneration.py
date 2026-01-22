from __future__ import annotations
import getpass
import os
import subprocess
import sys
from io import StringIO
from typing import Callable
from typing_extensions import NoReturn
from twisted.conch.test.keydata import (
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_keygeneration(self) -> None:
    self._testrun('ecdsa', '384')
    self._testrun('ecdsa', '384', privateKeySubtype='v1')
    self._testrun('ecdsa')
    self._testrun('ecdsa', privateKeySubtype='v1')
    self._testrun('ed25519')
    self._testrun('dsa', '2048')
    self._testrun('dsa', '2048', privateKeySubtype='v1')
    self._testrun('dsa')
    self._testrun('dsa', privateKeySubtype='v1')
    self._testrun('rsa', '2048')
    self._testrun('rsa', '2048', privateKeySubtype='v1')
    self._testrun('rsa')
    self._testrun('rsa', privateKeySubtype='v1')