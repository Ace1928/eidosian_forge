from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def hasPort(element):
    return not isLeaf(element) and element.attributes.get('port') == self.port