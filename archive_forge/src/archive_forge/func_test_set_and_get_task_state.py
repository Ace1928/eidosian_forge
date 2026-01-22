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
def test_set_and_get_task_state(self):
    s = self._get_storage()
    state = states.PENDING
    s.ensure_atom(test_utils.NoopTask('my task'))
    s.set_atom_state('my task', state)
    self.assertEqual(state, s.get_atom_state('my task'))