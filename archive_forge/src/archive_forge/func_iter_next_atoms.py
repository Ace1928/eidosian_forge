from concurrent import futures
import weakref
from automaton import machines
from oslo_utils import timeutils
from taskflow import logging
from taskflow import states as st
from taskflow.types import failure
from taskflow.utils import iter_utils
def iter_next_atoms(atom=None, apply_deciders=True):
    maybe_atoms_it = self._selector.iter_next_atoms(atom=atom)
    for atom, late_decider in maybe_atoms_it:
        if apply_deciders:
            proceed = late_decider.check_and_affect(self._runtime)
            if proceed:
                yield atom
        else:
            yield atom