import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_bad_rebind_args_value(self):
    self.assertRaises(TypeError, utils.TaskOneArg, rebind=object())