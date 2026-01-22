from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_defaultOutputs(self):
    """
        It is possible to write a transition with no outputs
        """

    class Mechanism(object):
        m = MethodicalMachine()

        @m.input()
        def finish(self):
            """final transition"""

        @m.state(initial=True)
        def start(self):
            """a start state"""

        @m.state()
        def finished(self):
            """a final state"""
        start.upon(finish, enter=finished)
    Mechanism().finish()