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
def sendBlock(self):
    block = self.file.read(self.blocksize)
    if block:
        self.transport.write(block)
        self.bytesSent = self.bytesSent + len(block)
    else:
        self.transport.loseConnection()
        self.completed = 1