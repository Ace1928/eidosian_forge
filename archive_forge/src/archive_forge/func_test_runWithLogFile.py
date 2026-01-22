import os
import socket
import subprocess
import sys
from itertools import count
from unittest import skipIf
from zope.interface import implementer
from twisted.conch.error import ConchError
from twisted.conch.test.keydata import (
from twisted.conch.test.test_ssh import ConchTestRealm
from twisted.cred import portal
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessExitedAlready
from twisted.internet.task import LoopingCall
from twisted.internet.utils import getProcessValue
from twisted.python import filepath, log, runtime
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
import sys, os
from twisted.conch.scripts.%s import run
def test_runWithLogFile(self):
    """
        It can store logs to a local file.
        """

    def cb_check_log(result):
        logContent = logPath.getContent()
        self.assertIn(b'Log opened.', logContent)
    logPath = filepath.FilePath(self.mktemp())
    d = self.execute(remoteCommand='echo goodbye', process=ConchTestOpenSSHProcess(), conchArgs=['--log', '--logfile', logPath.path, '--host-key-algorithms', 'ssh-rsa'])
    d.addCallback(self.assertEqual, b'goodbye\n')
    d.addCallback(cb_check_log)
    return d