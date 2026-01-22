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
def topicAuthor(self, user, channel, author, date):
    """
        Send the author of and time at which a topic was set for the given
        channel.

        This sends a 333 reply message, which is not part of the IRC RFC.

        @type user: C{str} or C{unicode}
        @param user: The user receiving the topic.  Only their nickname, not
            the full hostmask.

        @type channel: C{str} or C{unicode}
        @param channel: The channel for which this information is relevant.

        @type author: C{str} or C{unicode}
        @param author: The nickname (without hostmask) of the user who last set
            the topic.

        @type date: C{int}
        @param date: A POSIX timestamp (number of seconds since the epoch) at
            which the topic was last set.
        """
    self.sendLine(':%s %d %s %s %s %d' % (self.hostname, 333, user, channel, author, date))