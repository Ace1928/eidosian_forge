import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import tasks
def test_list_tasks_with_marker(self):
    tasks = self.controller.list(marker=_PENDING_ID, page_size=1)
    self.assertEqual(1, len(tasks))
    self.assertEqual(_PROCESSING_ID, tasks[0]['id'])