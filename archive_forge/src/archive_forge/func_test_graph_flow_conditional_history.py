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
def test_graph_flow_conditional_history(self):

    def even_odd_decider(history, allowed):
        total = sum(history.values())
        if total == allowed:
            return True
        return False
    flow = gf.Flow('root')
    task1 = utils.TaskMultiArgOneReturn(name='task1')
    task2 = utils.ProgressingTask(name='task2')
    task2_2 = utils.ProgressingTask(name='task2_2')
    task3 = utils.ProgressingTask(name='task3')
    task3_3 = utils.ProgressingTask(name='task3_3')
    flow.add(task1, task2, task2_2, task3, task3_3)
    flow.link(task1, task2, decider=functools.partial(even_odd_decider, allowed=2))
    flow.link(task2, task2_2)
    flow.link(task1, task3, decider=functools.partial(even_odd_decider, allowed=1))
    flow.link(task3, task3_3)
    engine = self._make_engine(flow)
    engine.storage.inject({'x': 0, 'y': 1, 'z': 1})
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        engine.run()
    expected = set(['task1.t RUNNING', 'task1.t SUCCESS(2)', 'task3.t IGNORE', 'task3_3.t IGNORE', 'task2.t RUNNING', 'task2.t SUCCESS(5)', 'task2_2.t RUNNING', 'task2_2.t SUCCESS(5)'])
    self.assertEqual(expected, set(capturer.values))
    engine = self._make_engine(flow)
    engine.storage.inject({'x': 0, 'y': 0, 'z': 1})
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        engine.run()
    expected = set(['task1.t RUNNING', 'task1.t SUCCESS(1)', 'task2.t IGNORE', 'task2_2.t IGNORE', 'task3.t RUNNING', 'task3.t SUCCESS(5)', 'task3_3.t RUNNING', 'task3_3.t SUCCESS(5)'])
    self.assertEqual(expected, set(capturer.values))