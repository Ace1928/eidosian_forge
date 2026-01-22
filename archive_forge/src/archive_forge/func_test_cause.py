import string
from taskflow import exceptions as exc
from taskflow import test
def test_cause(self):
    capture = None
    try:
        raise exc.TaskFlowException('broken', cause=IOError('dead'))
    except Exception as e:
        capture = e
    self.assertIsNotNone(capture)
    self.assertIsInstance(capture, exc.TaskFlowException)
    self.assertIsNotNone(capture.cause)
    self.assertIsInstance(capture.cause, IOError)