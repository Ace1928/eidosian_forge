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
def irc_ERR_NICKNAMEINUSE(self, prefix, params):
    """
        Called when we try to register or change to a nickname that is already
        taken.
        """
    self._attemptedNick = self.alterCollidedNick(self._attemptedNick)
    self.setNick(self._attemptedNick)