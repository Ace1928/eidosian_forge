from boto.compat import StringIO
from tests.compat import mock, unittest
from boto.glacier.job import Job
from boto.glacier.layer1 import Layer1
from boto.glacier.response import GlacierResponse
from boto.glacier.exceptions import TreeHashDoesNotMatchError
def test_calc_num_chunks(self):
    self.job.archive_size = 0
    self.assertEqual(self.job._calc_num_chunks(self.job.DefaultPartSize), 0)
    self.job.archive_size = 1
    self.assertEqual(self.job._calc_num_chunks(self.job.DefaultPartSize), 1)
    self.job.archive_size = self.job.DefaultPartSize + 1
    self.assertEqual(self.job._calc_num_chunks(self.job.DefaultPartSize), 2)