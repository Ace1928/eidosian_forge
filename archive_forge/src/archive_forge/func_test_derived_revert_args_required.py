import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_derived_revert_args_required(self):
    flow = utils.TaskRevertExtraArgs()
    engine = self._make_engine(flow)
    engine.storage.inject({'y': 4, 'z': 9, 'x': 17})
    self.assertRaises(exc.MissingDependencies, engine.run)
    engine.storage.inject({'revert_arg': None})
    self.assertRaises(exc.ExecutionFailure, engine.run)