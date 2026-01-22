from __future__ import annotations
import os
import sys
import time
import unittest as pyunit
import warnings
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
from zope.interface import implementer
from typing_extensions import TypeAlias
from twisted.python import log, reflect
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.util import untilConcludes
from twisted.trial import itrial, util
def upDownError(self, method, error, warn, printStatus):
    self._colorizer.write('  %s' % method, self.ERROR)
    if printStatus:
        self.endLine('[ERROR]', self.ERROR)
    super().upDownError(method, error, warn, printStatus)