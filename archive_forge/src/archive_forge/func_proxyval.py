from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def proxyval(self, visited):
    name = self.safe_name()
    tp_name = self.safe_tp_name()
    self_address = self.safe_self_addresss()
    return '<method-wrapper %s of %s object at %s>' % (name, tp_name, self_address)