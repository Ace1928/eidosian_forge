import collections
import contextlib
import functools
import threading
import futurist
import testtools
import taskflow.engines
from taskflow.engines.action_engine import engine as eng
from taskflow.engines.worker_based import engine as w_eng
from taskflow.engines.worker_based import worker as wkr
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow.persistence import models
from taskflow import states
from taskflow import task
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.types import graph as gr
from taskflow.utils import eventlet_utils as eu
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils as tu
def test_graph_flow_four_tasks_revert_failure(self):
    flow = gf.Flow('g-3-nasty').add(utils.NastyTask(name='task2', provides='b', requires=['a']), utils.FailingTask(name='task3', requires=['b']), utils.ProgressingTask(name='task1', provides='a'))
    engine = self._make_engine(flow)
    self.assertFailuresRegexp(RuntimeError, '^Gotcha', engine.run)
    self.assertEqual(states.FAILURE, engine.storage.get_flow_state())