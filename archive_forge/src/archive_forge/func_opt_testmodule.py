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
def opt_testmodule(self, filename):
    """
        Filename to grep for test cases (-*- test-case-name).
        """
    if not os.path.isfile(filename):
        sys.stderr.write(f"File {filename!r} doesn't exist\n")
        return
    filename = os.path.abspath(filename)
    if isTestFile(filename):
        self['tests'].append(filename)
    else:
        self['tests'].extend(getTestModules(filename))