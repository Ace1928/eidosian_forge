import socket
import sys
from typing import List, Optional, Sequence
from twisted import plugin
from twisted.application import strports
from twisted.application.service import MultiService
from twisted.cred import checkers, credentials, portal, strcred
from twisted.python import usage
from twisted.words import iwords, service
def opt_group(self, name):
    """Specify a group which should exist"""
    self['groups'].append(name.decode(sys.stdin.encoding))