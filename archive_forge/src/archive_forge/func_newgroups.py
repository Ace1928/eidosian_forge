import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def newgroups(self, date, *, file=None):
    """Process a NEWGROUPS command.  Arguments:
        - date: a date or datetime object
        Return:
        - resp: server response if successful
        - list: list of newsgroup names
        """
    if not isinstance(date, (datetime.date, datetime.date)):
        raise TypeError("the date parameter must be a date or datetime object, not '{:40}'".format(date.__class__.__name__))
    date_str, time_str = _unparse_datetime(date, self.nntp_version < 2)
    cmd = 'NEWGROUPS {0} {1}'.format(date_str, time_str)
    resp, lines = self._longcmdstring(cmd, file)
    return (resp, self._grouplist(lines))