import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_accumulateAdditionalOptions(self):
    """
        We pick up options that are only defined by having an
        appropriately named method on your Options class,
        e.g. def opt_foo(self, foo)
        """
    opts = FighterAceExtendedOptions()
    ag = _shellcomp.ZshArgumentsGenerator(opts, 'ace', BytesIO())
    self.assertIn('nocrash', ag.flagNameToDefinition)
    self.assertIn('nocrash', ag.allOptionsNameToDefinition)
    self.assertIn('difficulty', ag.paramNameToDefinition)
    self.assertIn('difficulty', ag.allOptionsNameToDefinition)