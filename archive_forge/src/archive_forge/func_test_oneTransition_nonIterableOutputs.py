from .._core import Automaton, NoTransition
from unittest import TestCase
def test_oneTransition_nonIterableOutputs(self):
    """
        L{Automaton.addTransition} raises a TypeError when given outputs
        that aren't iterable and doesn't add any transitions.
        """
    a = Automaton()
    nonIterableOutputs = 1
    self.assertRaises(TypeError, a.addTransition, 'fromState', 'viaSymbol', 'toState', nonIterableOutputs)
    self.assertFalse(a.inputAlphabet())
    self.assertFalse(a.outputAlphabet())
    self.assertFalse(a.states())
    self.assertFalse(a.allTransitions())