from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_requires_explicit_not_enough(self):
    self.assertRaisesRegex(ValueError, '^Missing arguments', MyTask, auto_extract=False, requires=('spam', 'eggs'))