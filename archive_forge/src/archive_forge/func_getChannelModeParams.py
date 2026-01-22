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
def getChannelModeParams(self):
    """
        Get channel modes that require parameters for correct parsing.

        @rtype: C{[str, str]}
        @return: C{[add, remove]}
        """
    params = ['', '']
    prefixes = self.supported.getFeature('PREFIX', {})
    params[0] = params[1] = ''.join(prefixes.keys())
    chanmodes = self.supported.getFeature('CHANMODES')
    if chanmodes is not None:
        params[0] += chanmodes.get('addressModes', '')
        params[0] += chanmodes.get('param', '')
        params[1] = params[0]
        params[0] += chanmodes.get('setParam', '')
    return params