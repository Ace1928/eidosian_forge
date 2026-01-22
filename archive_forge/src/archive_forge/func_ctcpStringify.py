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
def ctcpStringify(messages):
    """
    @type messages: a list of extended messages.  An extended
    message is a (tag, data) tuple, where 'data' may be L{None}, a
    string, or a list of strings to be joined with whitespace.

    @returns: String
    """
    coded_messages = []
    for tag, data in messages:
        if data:
            if not isinstance(data, str):
                try:
                    data = ' '.join(map(str, data))
                except TypeError:
                    pass
            m = f'{tag} {data}'
        else:
            m = str(tag)
        m = ctcpQuote(m)
        m = f'{X_DELIM}{m}{X_DELIM}'
        coded_messages.append(m)
    line = ''.join(coded_messages)
    return line