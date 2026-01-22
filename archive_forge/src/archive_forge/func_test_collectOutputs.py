from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_collectOutputs(self):
    """
        Outputs can be combined with the "collector" argument to "upon".
        """
    import operator

    class Machine(object):
        m = MethodicalMachine()

        @m.input()
        def input(self):
            """an input"""

        @m.output()
        def outputA(self):
            return 'A'

        @m.output()
        def outputB(self):
            return 'B'

        @m.state(initial=True)
        def state(self):
            """a state"""
        state.upon(input, state, [outputA, outputB], collector=lambda x: reduce(operator.add, x))
    m = Machine()
    self.assertEqual(m.input(), 'AB')