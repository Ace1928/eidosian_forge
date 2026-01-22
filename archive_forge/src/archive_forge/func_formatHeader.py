import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def formatHeader(self, records):
    """
        Return the header string for the specified records.
        """
    return ''