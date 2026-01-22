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
def dccAcceptResume(self, user, fileName, port, resumePos):
    """
        Send a DCC ACCEPT response to clients who have requested a resume.
        """
    self.ctcpMakeQuery(user, [('DCC', ['ACCEPT', fileName, port, resumePos])])