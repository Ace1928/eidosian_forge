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
def opt_help_orders(self):
    synopsis = 'Trial can attempt to run test cases and their methods in a few different orders. You can select any of the following options using --order=<foo>.\n'
    print(synopsis)
    for name, (description, _) in sorted(_runOrders.items()):
        print('   ', name, '\t', description)
    sys.exit(0)