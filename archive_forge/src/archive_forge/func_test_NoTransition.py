from .._core import Automaton, NoTransition
from unittest import TestCase
def test_NoTransition(self):
    """
        A L{NoTransition} exception describes the state and input symbol
        that caused it.
        """
    with self.assertRaises(TypeError):
        NoTransition()
    state = 'current-state'
    symbol = 'transitionless-symbol'
    noTransitionException = NoTransition(state=state, symbol=symbol)
    self.assertIs(noTransitionException.symbol, symbol)
    self.assertIn(state, str(noTransitionException))
    self.assertIn(symbol, str(noTransitionException))