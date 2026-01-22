import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_arguments_missing(self):
    flow = utils.TaskMultiArg()
    engine = self._make_engine(flow)
    engine.storage.inject({'a': 1, 'b': 4, 'x': 17})
    self.assertRaises(exc.MissingDependencies, engine.run)