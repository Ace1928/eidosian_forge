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
def test_save_fetch_revert(self):
    t = test_utils.GiveBackRevert('my task')
    s = self._get_storage()
    s.ensure_atom(t)
    s.set_atom_intention('my task', states.REVERT)
    s.save('my task', 2, state=states.REVERTED)
    self.assertRaises(exceptions.NotFound, s.get, 'my task')
    self.assertEqual(2, s.get_revert_result('my task'))