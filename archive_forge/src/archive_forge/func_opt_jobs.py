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
def opt_jobs(self, number):
    """
        Number of local workers to run, a strictly positive integer.
        """
    try:
        number = int(number)
    except ValueError:
        raise usage.UsageError("Expecting integer argument to jobs, got '%s'" % number)
    if number <= 0:
        raise usage.UsageError('Argument to jobs must be a strictly positive integer')
    self['jobs'] = number