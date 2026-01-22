import abc
import weakref
from oslo_utils import reflection
from oslo_utils import strutils
from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import executor as ex
from taskflow import logging
from taskflow import retry as retry_atom
from taskflow import states as st
def complete_failure(self, node, outcome, failure):
    """Performs post-execution completion of a nodes failure.

        Returns whether the result should be saved into an accumulator of
        failures or whether this should not be done.
        """
    if outcome == ex.EXECUTED and self._resolve:
        self._process_atom_failure(node, failure)
        return False
    else:
        return True