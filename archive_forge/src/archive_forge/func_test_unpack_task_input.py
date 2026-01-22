from unittest import mock
import urllib
from glance.common import exception
from glance.common.scripts import utils as script_utils
import glance.tests.utils as test_utils
def test_unpack_task_input(self):
    task_input = {'import_from': 'foo', 'import_from_format': 'bar', 'image_properties': 'baz'}
    task = mock.Mock(task_input=task_input)
    self.assertEqual(task_input, script_utils.unpack_task_input(task))