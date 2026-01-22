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
@fasteners.write_locked
def set_atom_intention(self, atom_name, intention):
    """Sets the intention of an atom given an atoms name."""
    source, clone = self._atomdetail_by_name(atom_name, clone=True)
    if source.intention != intention:
        clone.intention = intention
        self._with_connection(self._save_atom_detail, source, clone)