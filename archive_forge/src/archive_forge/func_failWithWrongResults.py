import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def failWithWrongResults():
    return [3, 5, 9]