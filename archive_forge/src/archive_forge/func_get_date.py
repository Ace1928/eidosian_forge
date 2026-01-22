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
def get_date(self):
    """Return delivery date of message, in seconds since the epoch."""
    return self._date