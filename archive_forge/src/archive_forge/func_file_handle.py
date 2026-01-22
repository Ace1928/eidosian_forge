from __future__ import print_function
import errno
import logging
import os
import time
from oauth2client import util
def file_handle(self):
    """Return the file_handle to the opened file."""
    return self._opener.file_handle()