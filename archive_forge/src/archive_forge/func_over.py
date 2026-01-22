import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def over(self, message_spec, *, file=None):
    """Process an OVER command.  If the command isn't supported, fall
        back to XOVER. Arguments:
        - message_spec:
            - either a message id, indicating the article to fetch
              information about
            - or a (start, end) tuple, indicating a range of article numbers;
              if end is None, information up to the newest message will be
              retrieved
            - or None, indicating the current article number must be used
        - file: Filename string or file object to store the result in
        Returns:
        - resp: server response if successful
        - list: list of dicts containing the response fields

        NOTE: the "message id" form isn't supported by XOVER
        """
    cmd = 'OVER' if 'OVER' in self._caps else 'XOVER'
    if isinstance(message_spec, (tuple, list)):
        start, end = message_spec
        cmd += ' {0}-{1}'.format(start, end or '')
    elif message_spec is not None:
        cmd = cmd + ' ' + message_spec
    resp, lines = self._longcmdstring(cmd, file)
    fmt = self._getoverviewfmt()
    return (resp, _parse_overview(lines, fmt))