import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
@fasteners.read_locked
def get_atoms_states(self, atom_names):
    """Gets a dict of atom name => (state, intention) given atom names."""
    details = {}
    for name in set(atom_names):
        source, _clone = self._atomdetail_by_name(name)
        details[name] = (source.state, source.intention)
    return details