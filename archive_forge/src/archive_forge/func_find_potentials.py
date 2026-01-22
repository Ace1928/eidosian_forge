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
def find_potentials(self, looking_for, scope_walker=None):
    """Returns the accessible **potential** providers."""
    _searched_providers, providers_and_results = self._find(looking_for, scope_walker=scope_walker, short_circuit=False, find_potentials=True)
    return set((p for p, _provider_results in providers_and_results))