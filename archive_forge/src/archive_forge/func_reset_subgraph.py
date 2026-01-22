import collections
import functools
from futurist import waiters
from taskflow import deciders as de
from taskflow.engines.action_engine.actions import retry as ra
from taskflow.engines.action_engine.actions import task as ta
from taskflow.engines.action_engine import builder as bu
from taskflow.engines.action_engine import compiler as com
from taskflow.engines.action_engine import completer as co
from taskflow.engines.action_engine import scheduler as sched
from taskflow.engines.action_engine import scopes as sc
from taskflow.engines.action_engine import selector as se
from taskflow.engines.action_engine import traversal as tr
from taskflow import exceptions as exc
from taskflow import logging
from taskflow import states as st
from taskflow.utils import misc
from taskflow.flow import (LINK_DECIDER, LINK_DECIDER_DEPTH)  # noqa
def reset_subgraph(self, atom, state=st.PENDING, intention=st.EXECUTE):
    """Resets a atoms subgraph to the given state and intention.

        The subgraph is contained of **all** of the atoms successors.
        """
    execution_graph = self._compilation.execution_graph
    atoms_it = tr.depth_first_iterate(execution_graph, atom, tr.Direction.FORWARD)
    return self.reset_atoms(atoms_it, state=state, intention=intention)