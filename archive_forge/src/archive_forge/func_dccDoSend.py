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
def dccDoSend(self, user, address, port, fileName, size, data):
    """
        Called when I receive a DCC SEND offer from a client.

        By default, I do nothing here.

        @param user: The hostmask of the requesting user.
        @type user: L{bytes}

        @param address: The IP address of the requesting user.
        @type address: L{bytes}

        @param port: An integer representing the port of the requesting user.
        @type port: L{int}

        @param fileName: The name of the file to be transferred.
        @type fileName: L{bytes}

        @param size: The size of the file to be transferred, which may be C{-1}
            if the size of the file was not specified in the DCC SEND request.
        @type size: L{int}

        @param data: A 3-list of [fileName, address, port].
        @type data: L{list}
        """