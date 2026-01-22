import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
def whois(self, nickname, server=None):
    """
        Retrieve user information about the given nickname.

        @type nickname: C{str}
        @param nickname: The nickname about which to retrieve information.

        @since: 8.2
        """
    if server is None:
        self.sendLine('WHOIS ' + nickname)
    else:
        self.sendLine(f'WHOIS {server} {nickname}')