import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def makeTest(cmdName, optionsFQPN):

    def runTest(self):
        return test_genZshFunction(self, cmdName, optionsFQPN)
    return runTest