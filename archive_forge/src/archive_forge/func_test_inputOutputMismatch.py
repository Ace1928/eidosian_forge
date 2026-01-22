from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_inputOutputMismatch(self):
    """
        All the argument lists of the outputs for a given input must match; if
        one does not the call to C{upon} will raise a C{TypeError}.
        """

    class Mechanism(object):
        m = MethodicalMachine()

        @m.input()
        def nameOfInput(self, a):
            """an input"""

        @m.output()
        def outputThatMatches(self, a):
            """an output that matches"""

        @m.output()
        def outputThatDoesntMatch(self, b):
            """an output that doesn't match"""

        @m.state()
        def state(self):
            """a state"""
        with self.assertRaises(TypeError) as cm:
            state.upon(nameOfInput, state, [outputThatMatches, outputThatDoesntMatch])
        self.assertIn('nameOfInput', str(cm.exception))
        self.assertIn('outputThatDoesntMatch', str(cm.exception))