import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def checkOpenfunCalled(self, conn=None):
    if not conn:
        self.assertTrue(self.openfun_called)
    else:
        self.assertIn(conn, self.openfun_called)