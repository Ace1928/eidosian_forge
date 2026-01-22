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
def irc_PRIVMSG(self, prefix, params):
    """
        Called when we get a message.
        """
    user = prefix
    channel = params[0]
    message = params[-1]
    if not message:
        return
    if message[0] == X_DELIM:
        m = ctcpExtract(message)
        if m['extended']:
            self.ctcpQuery(user, channel, m['extended'])
        if not m['normal']:
            return
        message = ' '.join(m['normal'])
    self.privmsg(user, channel, message)