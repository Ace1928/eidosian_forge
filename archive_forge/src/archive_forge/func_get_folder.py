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
def get_folder(self, folder):
    """Return an MH instance for the named folder."""
    return MH(os.path.join(self._path, folder), factory=self._factory, create=False)