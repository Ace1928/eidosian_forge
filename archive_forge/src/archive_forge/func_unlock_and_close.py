from __future__ import print_function
import errno
import logging
import os
import time
from oauth2client import util
def unlock_and_close(self):
    """Unlock and close a file."""
    self._opener.unlock_and_close()