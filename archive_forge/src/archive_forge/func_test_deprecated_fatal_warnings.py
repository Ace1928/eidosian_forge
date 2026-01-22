from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import validator
@mock.patch('oslo_config.validator.load_opt_data')
def test_deprecated_fatal_warnings(self, mock_lod):
    mock_lod.return_value = OPT_DATA
    self.conf_fixture.config(opt_data='mocked.yaml', input_file='mocked.conf', fatal_warnings=True)
    m = mock.mock_open(read_data=DEPRECATED_CONF)
    with mock.patch('builtins.open', m):
        self.assertEqual(1, validator._validate(self.conf))