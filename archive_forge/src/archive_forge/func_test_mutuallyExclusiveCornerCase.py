import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_mutuallyExclusiveCornerCase(self):
    """
        Exercise a corner-case of ZshArgumentsGenerator.makeExcludesDict()
        where the long option name already exists in the `excludes` dict being
        built.
        """

    class OddFighterAceOptions(FighterAceExtendedOptions):
        optFlags = [['anatra', None, 'Select the Anatra DS as your dogfighter aircraft']]
        compData = Completions(mutuallyExclusive=[['anatra', 'fokker', 'albatros', 'spad', 'bristol']])
    opts = OddFighterAceOptions()
    ag = _shellcomp.ZshArgumentsGenerator(opts, 'ace', BytesIO())
    expected = {'albatros': {'anatra', 'b', 'bristol', 'f', 'fokker', 's', 'spad'}, 'anatra': {'a', 'albatros', 'b', 'bristol', 'f', 'fokker', 's', 'spad'}, 'bristol': {'a', 'albatros', 'anatra', 'f', 'fokker', 's', 'spad'}, 'fokker': {'a', 'albatros', 'anatra', 'b', 'bristol', 's', 'spad'}, 'spad': {'a', 'albatros', 'anatra', 'b', 'bristol', 'f', 'fokker'}}
    self.assertEqual(ag.excludes, expected)