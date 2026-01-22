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
def fetch_all(self, many_handler=None):
    """Fetch all named ``execute`` results known so far."""

    def _many_handler(values):
        if len(values) > 1:
            return values
        return values[0]
    if many_handler is None:
        many_handler = _many_handler
    results = {}
    for name in self._reverse_mapping.keys():
        try:
            results[name] = self.fetch(name, many_handler=many_handler)
        except exceptions.NotFound:
            pass
    return results