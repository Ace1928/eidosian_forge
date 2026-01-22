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
def parsemsg(s):
    """
    Breaks a message from an IRC server into its prefix, command, and
    arguments.

    @param s: The message to break.
    @type s: L{bytes}

    @return: A tuple of (prefix, command, args).
    @rtype: L{tuple}
    """
    prefix = ''
    trailing = []
    if not s:
        raise IRCBadMessage('Empty line.')
    if s[0:1] == ':':
        prefix, s = s[1:].split(' ', 1)
    if s.find(' :') != -1:
        s, trailing = s.split(' :', 1)
        args = s.split()
        args.append(trailing)
    else:
        args = s.split()
    command = args.pop(0)
    return (prefix, command, args)