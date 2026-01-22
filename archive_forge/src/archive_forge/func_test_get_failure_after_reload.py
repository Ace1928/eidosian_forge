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
def test_get_failure_after_reload(self):
    a_failure = failure.Failure.from_exception(RuntimeError('Woot!'))
    s = self._get_storage()
    s.ensure_atom(test_utils.NoopTask('my task'))
    s.save('my task', a_failure, states.FAILURE)
    s2 = self._get_storage(s._flowdetail)
    self.assertTrue(s2.has_failures())
    self.assertEqual(1, len(s2.get_failures()))
    self.assertTrue(a_failure.matches(s2.get('my task')))
    self.assertEqual(states.FAILURE, s2.get_atom_state('my task'))