import os
from unittest import mock
import dogpile.cache
from ironicclient.common import filecache
from ironicclient.tests.unit import utils
@mock.patch.object(os.path, 'isfile', autospec=True)
def test_retrieve_data_no_cache_file(self, mock_isfile):
    mock_isfile.return_value = False
    self.assertIsNone(filecache.retrieve_data(host='spam', port='eggs'))