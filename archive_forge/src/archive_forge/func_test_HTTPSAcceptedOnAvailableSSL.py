from __future__ import annotations
import os
import stat
from typing import cast
from unittest import skipIf
from twisted.internet import endpoints, reactor
from twisted.internet.interfaces import IReactorCore, IReactorUNIX
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.python.threadpool import ThreadPool
from twisted.python.usage import UsageError
from twisted.spread.pb import PBServerFactory
from twisted.trial.unittest import TestCase
from twisted.web import demo
from twisted.web.distrib import ResourcePublisher, UserDirectory
from twisted.web.script import PythonScript
from twisted.web.server import Site
from twisted.web.static import Data, File
from twisted.web.tap import (
from twisted.web.test.requesthelper import DummyRequest
from twisted.web.twcgi import CGIScript
from twisted.web.wsgi import WSGIResource
@skipIf(requireModule('OpenSSL.SSL') is None, 'SSL module is not available.')
def test_HTTPSAcceptedOnAvailableSSL(self) -> None:
    """
        When SSL support is present, it accepts the --https option.
        """
    options = Options()
    options.parseOptions(['--https=443'])
    self.assertIn('ssl', options['ports'][0])
    self.assertIn('443', options['ports'][0])