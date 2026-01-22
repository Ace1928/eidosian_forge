from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase

        L{crefutil._InstanceMethod} raises L{AssertionError} to indicate it
        should not be called.  This should not be possible with any of its API
        clients, but is provided for helping to debug.
        