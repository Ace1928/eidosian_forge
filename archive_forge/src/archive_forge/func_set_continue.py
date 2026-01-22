import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
def set_continue(self):
    if self.__debugger_used:
        pdb.Pdb.set_continue(self)