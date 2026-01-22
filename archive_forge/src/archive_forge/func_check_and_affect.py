import abc
import itertools
from taskflow import deciders
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states
def check_and_affect(self, runtime):
    """Handles :py:func:`~.tally` + :py:func:`~.affect` in right order.

        NOTE(harlowja):  If there are zero 'nay' edge deciders then it is
        assumed this decider should allow running.

        Returns boolean of whether this decider allows for running (or not).
        """
    nay_voters = self.tally(runtime)
    if nay_voters:
        self.affect(runtime, nay_voters)
        return False
    return True