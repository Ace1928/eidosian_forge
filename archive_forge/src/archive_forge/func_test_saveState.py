from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_saveState(self):
    """
        L{MethodicalMachine.serializer} is a decorator that modifies its
        decoratee's signature to take a "state" object as its first argument,
        which is the "serialized" argument to the L{MethodicalMachine.state}
        decorator.
        """

    class Mechanism(object):
        m = MethodicalMachine()

        def __init__(self):
            self.value = 1

        @m.state(serialized='first-state', initial=True)
        def first(self):
            """First state."""

        @m.state(serialized='second-state')
        def second(self):
            """Second state."""

        @m.serializer()
        def save(self, state):
            return {'machine-state': state, 'some-value': self.value}
    self.assertEqual(Mechanism().save(), {'machine-state': 'first-state', 'some-value': 1})