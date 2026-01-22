import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def formatFooter(self, records):
    """
        Return the footer string for the specified records.
        """
    return ''