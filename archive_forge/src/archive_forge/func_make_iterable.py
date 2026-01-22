import getpass
import io
import logging
import urllib.parse
import urllib.request
from distutils._log import log
from warnings import warn
from .._itertools import always_iterable
from ..core import PyPIRCCommand
def make_iterable(values):
    if values is None:
        return [None]
    return always_iterable(values)