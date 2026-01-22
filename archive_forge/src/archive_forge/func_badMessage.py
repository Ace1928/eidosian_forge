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
def badMessage(self, line, excType, excValue, tb):
    """
        When I get a message that's so broken I can't use it.

        @param line: The indecipherable message.
        @type line: L{bytes}

        @param excType: The exception type of the exception raised by the
            message.
        @type excType: L{type}

        @param excValue: The exception parameter of excType or its associated
            value(the second argument to C{raise}).
        @type excValue: L{BaseException}

        @param tb: The Traceback as a traceback object.
        @type tb: L{traceback}
        """
    log.msg(line)
    log.msg(''.join(traceback.format_exception(excType, excValue, tb)))