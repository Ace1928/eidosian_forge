import gc
import inspect
import os
import pdb
import random
import sys
import time
import trace
import warnings
from typing import NoReturn, Optional, Type
from twisted import plugin
from twisted.application import app
from twisted.internet import defer
from twisted.python import failure, reflect, usage
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedModule
from twisted.trial import itrial, runner
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial.unittest import TestSuite
def opt_recursionlimit(self, arg):
    """
        see sys.setrecursionlimit()
        """
    try:
        sys.setrecursionlimit(int(arg))
    except (TypeError, ValueError):
        raise usage.UsageError('argument to recursionlimit must be an integer')
    else:
        self['recursionlimit'] = int(arg)