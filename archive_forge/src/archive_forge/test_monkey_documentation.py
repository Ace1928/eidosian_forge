from __future__ import annotations
from typing_extensions import NoReturn
from twisted.python.monkey import MonkeyPatcher
from twisted.trial import unittest

        Exceptions propagate through the L{MonkeyPatcher} context-manager
        exit method.
        