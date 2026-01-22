import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def newnews(self, group, date, *, file=None):
    """Process a NEWNEWS command.  Arguments:
        - group: group name or '*'
        - date: a date or datetime object
        Return:
        - resp: server response if successful
        - list: list of message ids
        """
    if not isinstance(date, (datetime.date, datetime.date)):
        raise TypeError("the date parameter must be a date or datetime object, not '{:40}'".format(date.__class__.__name__))
    date_str, time_str = _unparse_datetime(date, self.nntp_version < 2)
    cmd = 'NEWNEWS {0} {1} {2}'.format(group, date_str, time_str)
    return self._longcmdstring(cmd, file)