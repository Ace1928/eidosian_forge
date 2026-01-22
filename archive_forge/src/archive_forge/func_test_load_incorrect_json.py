import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(builtins, 'open', mock.mock_open(read_data='{{bbb'))
def test_load_incorrect_json(self):
    fname = 'abc.json'
    self.assertRaisesRegex(exc.ClientException, 'File "%s" is invalid' % fname, create_resources.load_from_file, fname)