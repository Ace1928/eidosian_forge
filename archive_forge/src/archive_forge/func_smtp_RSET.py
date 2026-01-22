import sys
import os
import errno
import getopt
import time
import socket
import collections
from warnings import _deprecated, warn
from email._header_value_parser import get_addr_spec, get_angle_addr
import asyncore
import asynchat
def smtp_RSET(self, arg):
    if arg:
        self.push('501 Syntax: RSET')
        return
    self._set_rset_state()
    self.push('250 OK')