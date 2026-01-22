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
def sendMessage(self, command, *parameter_list, **prefix):
    """
        Send a line formatted as an IRC message.

        First argument is the command, all subsequent arguments are parameters
        to that command.  If a prefix is desired, it may be specified with the
        keyword argument 'prefix'.

        The L{sendCommand} method is generally preferred over this one.
        Notably, this method does not support sending message tags, while the
        L{sendCommand} method does.
        """
    if not command:
        raise ValueError('IRC message requires a command.')
    if ' ' in command or command[0] == ':':
        raise ValueError("Somebody screwed up, 'cuz this doesn't look like a command to me: %s" % command)
    line = ' '.join([command] + list(parameter_list))
    if 'prefix' in prefix:
        line = ':{} {}'.format(prefix['prefix'], line)
    self.sendLine(line)
    if len(parameter_list) > 15:
        log.msg('Message has %d parameters (RFC allows 15):\n%s' % (len(parameter_list), line))