import os
import re
import sys
from getopt import getopt, GetoptError
from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error
import typing as t
def default_option(self, fn, optstr):
    """Make an entry in the options_table for fn, with value optstr"""
    if fn not in self.lsmagic():
        error('%s is not a magic function' % fn)
    self.options_table[fn] = optstr