import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_poorlyDescribedOptMethod(self):
    """
        Test corner case fetching an option description from a method docstring
        """
    opts = FighterAceOptions()
    argGen = _shellcomp.ZshArgumentsGenerator(opts, 'ace', None)
    descr = argGen.getDescription('silly')
    self.assertEqual(descr, 'silly')