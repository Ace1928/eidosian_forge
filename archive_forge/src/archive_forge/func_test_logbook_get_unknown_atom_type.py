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
def test_logbook_get_unknown_atom_type(self):
    self.assertRaisesRegex(TypeError, 'Unknown atom', models.atom_detail_class, 'some_detail')