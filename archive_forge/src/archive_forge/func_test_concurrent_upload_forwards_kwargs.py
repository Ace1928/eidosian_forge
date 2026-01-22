from boto.compat import StringIO
from tests.compat import mock, unittest
from boto.glacier import vault
from boto.glacier.job import Job
from boto.glacier.response import GlacierResponse
def test_concurrent_upload_forwards_kwargs(self):
    v = vault.Vault(None, None)
    with mock.patch('boto.glacier.vault.ConcurrentUploader') as c:
        c.return_value.upload.return_value = 'archive_id'
        archive_id = v.concurrent_create_archive_from_file('filename', 'my description', num_threads=10, part_size=1024 * 1024 * 1024 * 8)
        c.assert_called_with(None, None, num_threads=10, part_size=1024 * 1024 * 1024 * 8)