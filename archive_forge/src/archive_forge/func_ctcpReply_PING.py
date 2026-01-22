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
def ctcpReply_PING(self, user, channel, data):
    nick = user.split('!', 1)[0]
    if not self._pings or (nick, data) not in self._pings:
        raise IRCBadMessage(f'Bogus PING response from {user}: {data}')
    t0 = self._pings[nick, data]
    self.pong(user, time.time() - t0)