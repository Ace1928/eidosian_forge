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
def dcc_ACCEPT(self, user, channel, data):
    data = shlex.split(data)
    if len(data) < 3:
        raise IRCBadMessage(f'malformed DCC SEND ACCEPT request: {data!r}')
    filename, port, resumePos = data[:3]
    try:
        port = int(port)
        resumePos = int(resumePos)
    except ValueError:
        return
    self.dccDoAcceptResume(user, filename, port, resumePos)