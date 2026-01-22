from boto.compat import StringIO
from tests.compat import mock, unittest
from boto.glacier import vault
from boto.glacier.job import Job
from boto.glacier.response import GlacierResponse
def test_part_size_needs_to_be_adjusted(self):
    self.getsize.return_value = 400 * 1024 * 1024 * 1024
    self.vault.create_archive_writer = mock.Mock()
    with mock.patch('boto.glacier.vault.open', self.mock_open, create=True):
        self.vault.create_archive_from_file('myfile')
    expected_part_size = 64 * 1024 * 1024
    self.vault.create_archive_writer.assert_called_with(description=mock.ANY, part_size=expected_part_size)