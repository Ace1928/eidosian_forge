import operator
import weakref
from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import deciders
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states as st
from taskflow.utils import iter_utils
def iter_connected_states():
    for atom in connected_fetcher():
        atom_states = self._storage.get_atoms_states([atom.name])
        yield (atom, atom_states[atom.name])