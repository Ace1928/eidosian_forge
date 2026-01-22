from boto.compat import StringIO
from tests.compat import mock, unittest
from boto.glacier.job import Job
from boto.glacier.layer1 import Layer1
from boto.glacier.response import GlacierResponse
from boto.glacier.exceptions import TreeHashDoesNotMatchError
def test_download_to_fileobj(self):
    http_response = mock.Mock(read=mock.Mock(return_value='xyz'))
    response = GlacierResponse(http_response, None)
    response['TreeHash'] = 'tree_hash'
    self.api.get_job_output.return_value = response
    fileobj = StringIO()
    self.job.archive_size = 3
    with mock.patch('boto.glacier.job.tree_hash_from_str') as t:
        t.return_value = 'tree_hash'
        self.job.download_to_fileobj(fileobj)
    fileobj.seek(0)
    self.assertEqual(http_response.read.return_value, fileobj.read())