from __future__ import annotations
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.trial.unittest import TestCase

        L{SSHChannel.getHost} returns the same object as the underlying
        transport's C{getHost} method returns.
        