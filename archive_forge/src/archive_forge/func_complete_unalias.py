import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def complete_unalias(self, text, line, begidx, endidx):
    return [a for a in self.aliases if a.startswith(text)]