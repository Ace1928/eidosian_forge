import tempfile
from boto.compat import Queue
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.concurrent import ConcurrentUploader, ConcurrentDownloader
from boto.glacier.concurrent import UploadWorkerThread
from boto.glacier.concurrent import _END_SENTINEL
def test_num_retries_is_obeyed(self):
    api = mock.Mock()
    job_queue = Queue()
    result_queue = Queue()
    upload_thread = UploadWorkerThread(api, 'vault_name', self.filename, 'upload_id', job_queue, result_queue, num_retries=2, time_between_retries=0)
    api.upload_part.side_effect = Exception()
    job_queue.put((0, 1024))
    job_queue.put(_END_SENTINEL)
    upload_thread.run()
    self.assertEqual(api.upload_part.call_count, 3)