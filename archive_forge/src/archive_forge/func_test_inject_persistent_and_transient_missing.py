import contextlib
import threading
from oslo_utils import uuidutils
from taskflow import exceptions
from taskflow.persistence import backends
from taskflow.persistence import models
from taskflow import states
from taskflow import storage
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
from taskflow.utils import persistence_utils as p_utils
def test_inject_persistent_and_transient_missing(self):
    t = test_utils.ProgressingTask('my retry', requires=['x'])
    s = self._get_storage()
    s.ensure_atom(t)
    missing = s.fetch_unsatisfied_args(t.name, t.rebind)
    self.assertEqual(set(['x']), missing)
    s.inject_atom_args(t.name, {'x': 2}, transient=False)
    s.inject_atom_args(t.name, {'x': 3}, transient=True)
    missing = s.fetch_unsatisfied_args(t.name, t.rebind)
    self.assertEqual(set(), missing)
    args = s.fetch_mapped_args(t.rebind, atom_name=t.name)
    self.assertEqual(3, args['x'])