import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def xhdr(self, hdr, str, *, file=None):
    """Process an XHDR command (optional server extension).  Arguments:
        - hdr: the header type (e.g. 'subject')
        - str: an article nr, a message id, or a range nr1-nr2
        - file: Filename string or file object to store the result in
        Returns:
        - resp: server response if successful
        - list: list of (nr, value) strings
        """
    pat = re.compile('^([0-9]+) ?(.*)\n?')
    resp, lines = self._longcmdstring('XHDR {0} {1}'.format(hdr, str), file)

    def remove_number(line):
        m = pat.match(line)
        return m.group(1, 2) if m else line
    return (resp, [remove_number(line) for line in lines])