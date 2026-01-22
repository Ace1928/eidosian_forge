import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import tasks
def test_list_tasks_with_invalid_sort_dir(self):
    self.assertRaises(ValueError, self.controller.list, sort_dir='invalid')