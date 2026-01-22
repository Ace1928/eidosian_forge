import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
def remove_flag(self, flag):
    """Unset the given string flag(s) without changing others."""
    if 'Status' in self or 'X-Status' in self:
        self.set_flags(''.join(set(self.get_flags()) - set(flag)))