import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(builtins, 'open', autospec=True)
def test_load_ioerror(self, mock_open):
    mock_open.side_effect = IOError('file does not exist')
    fname = 'abc.json'
    self.assertRaisesRegex(exc.ClientException, 'Cannot read file', create_resources.load_from_file, fname)