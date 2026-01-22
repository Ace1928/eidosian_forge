import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import tasks
def test_list_tasks_with_wrong_status(self):
    filters = {'filters': {'status': 'fake'}}
    tasks = self.controller.list(**filters)
    self.assertEqual(0, len(tasks))