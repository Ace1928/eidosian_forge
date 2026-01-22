import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def setLoggerClass(self, klass):
    """
        Set the class to be used when instantiating a logger with this Manager.
        """
    if klass != Logger:
        if not issubclass(klass, Logger):
            raise TypeError('logger not derived from logging.Logger: ' + klass.__name__)
    self.loggerClass = klass