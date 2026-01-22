import contextlib
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import task
from taskflow import test
from taskflow.utils import persistence_utils as p_utils
def test_storage_progress_detail(self):
    flo = ProgressTaskWithDetails('test')
    e = self._make_engine(flo)
    e.run()
    end_progress = e.storage.get_task_progress('test')
    self.assertEqual(1.0, end_progress)
    end_details = e.storage.get_task_progress_details('test')
    self.assertEqual(0.5, end_details.get('at_progress'))
    self.assertEqual({'test': 'test data', 'foo': 'bar'}, end_details.get('details'))