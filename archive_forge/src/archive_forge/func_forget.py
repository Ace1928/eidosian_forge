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
def forget(self):
    self.lineno = None
    self.stack = []
    self.curindex = 0
    self.curframe = None
    self.tb_lineno.clear()