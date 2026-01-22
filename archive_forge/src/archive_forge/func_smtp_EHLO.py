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
def smtp_EHLO(self, arg):
    if not arg:
        self.push('501 Syntax: EHLO hostname')
        return
    if self.seen_greeting:
        self.push('503 Duplicate HELO/EHLO')
        return
    self._set_rset_state()
    self.seen_greeting = arg
    self.extended_smtp = True
    self.push('250-%s' % self.fqdn)
    if self.data_size_limit:
        self.push('250-SIZE %s' % self.data_size_limit)
        self.command_size_limits['MAIL'] += 26
    if not self._decode_data:
        self.push('250-8BITMIME')
    if self.enable_SMTPUTF8:
        self.push('250-SMTPUTF8')
        self.command_size_limits['MAIL'] += 10
    self.push('250 HELP')