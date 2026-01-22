import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
@mock.patch.object(builtins, 'open', autospec=True)
def test_handle_json_from_file_open_fail(self, mock_open):
    mock_file_object = mock.MagicMock()
    mock_file_handle = mock.MagicMock()
    mock_file_handle.__enter__.return_value = mock_file_object
    mock_open.return_value = mock_file_handle
    mock_file_object.read.side_effect = IOError
    with tempfile.NamedTemporaryFile(mode='w') as f:
        self.assertRaisesRegex(exc.InvalidAttribute, 'from file', utils.handle_json_from_file, f.name)
        mock_open.assert_called_once_with(f.name, 'r')
        mock_file_object.read.assert_called_once_with()