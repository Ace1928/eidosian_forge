import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def setLevel(self, level):
    """
        Set the specified level on the underlying logger.
        """
    self.logger.setLevel(level)