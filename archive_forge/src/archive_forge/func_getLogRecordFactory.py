import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def getLogRecordFactory():
    """
    Return the factory to be used when instantiating a log record.
    """
    return _logRecordFactory