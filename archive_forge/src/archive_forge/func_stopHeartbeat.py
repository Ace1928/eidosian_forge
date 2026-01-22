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
def stopHeartbeat(self):
    """
        Stop sending I{PING} messages to keep the connection to the server
        alive.

        @since: 11.1
        """
    if self._heartbeat is not None:
        self._heartbeat.stop()
        self._heartbeat = None