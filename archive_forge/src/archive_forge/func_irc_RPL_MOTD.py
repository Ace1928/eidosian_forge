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
def irc_RPL_MOTD(self, prefix, params):
    if params[-1].startswith('- '):
        params[-1] = params[-1][2:]
    if self.motd is None:
        self.motd = []
    self.motd.append(params[-1])