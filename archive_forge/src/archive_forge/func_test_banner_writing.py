import io
from oslo_utils import reflection
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import worker
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
def test_banner_writing(self):
    buf = io.StringIO()
    w = self.worker()
    w.run(banner_writer=buf.write)
    w.wait()
    w.stop()
    self.assertGreater(0, len(buf.getvalue()))