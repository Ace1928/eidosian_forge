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
def test_get_failure_from_reverted_task(self):
    a_failure = failure.Failure.from_exception(RuntimeError('Woot!'))
    s = self._get_storage()
    s.ensure_atom(test_utils.NoopTask('my task'))
    s.save('my task', a_failure, states.FAILURE)
    s.set_atom_state('my task', states.REVERTING)
    self.assertEqual(a_failure, s.get('my task'))
    s.set_atom_state('my task', states.REVERTED)
    self.assertEqual(a_failure, s.get('my task'))