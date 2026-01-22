from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_bad_provides(self):
    self.assertRaisesRegex(TypeError, '^Atom provides', MyTask, provides=object())