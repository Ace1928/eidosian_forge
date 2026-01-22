import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def hideResourceModule(self):
    """
        Make the L{resource} module unimportable for the remainder of the
        current test method.
        """
    sys.modules['resource'] = None