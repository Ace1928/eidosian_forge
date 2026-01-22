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
def ctcpUnknownQuery(self, user, channel, tag, data):
    """
        Fallback handler for unrecognized CTCP queries.

        No CTCP I{ERRMSG} reply is made to remove a potential denial of service
        avenue.
        """
    log.msg(f'Unknown CTCP query from {user!r}: {tag!r} {data!r}')