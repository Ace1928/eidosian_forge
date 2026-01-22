import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def slave(self):
    """Process a SLAVE command.  Returns:
        - resp: server response if successful
        """
    return self._shortcmd('SLAVE')