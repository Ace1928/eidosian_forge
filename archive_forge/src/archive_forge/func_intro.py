import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def intro(self):
    self.output.write('Welcome to Python {0}\'s help utility! If this is your first time using\nPython, you should definitely check out the tutorial at\nhttps://docs.python.org/{0}/tutorial/.\n\nEnter the name of any module, keyword, or topic to get help on writing\nPython programs and using Python modules.  To get a list of available\nmodules, keywords, symbols, or topics, enter "modules", "keywords",\n"symbols", or "topics".\n\nEach module also comes with a one-line summary of what it does; to list\nthe modules whose name or summary contain a given string such as "spam",\nenter "modules spam".\n\nTo quit this help utility and return to the interpreter,\nenter "q" or "quit".\n'.format('%d.%d' % sys.version_info[:2]))