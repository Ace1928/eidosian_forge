import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import tasks
def test_list_tasks_paginated(self):
    tasks = self.controller.list(page_size=1)
    self.assertEqual(_PENDING_ID, tasks[0].id)
    self.assertEqual('import', tasks[0].type)
    self.assertEqual(_PROCESSING_ID, tasks[1].id)
    self.assertEqual('import', tasks[1].type)