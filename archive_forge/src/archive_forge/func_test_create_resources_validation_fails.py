import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(create_resources, 'create_nodes', autospec=True)
@mock.patch.object(create_resources, 'create_chassis', autospec=True)
@mock.patch.object(jsonschema, 'validate', side_effect=jsonschema.ValidationError(''), autospec=True)
@mock.patch.object(create_resources, 'load_from_file', side_effect=[schema_pov_invalid_json], autospec=True)
def test_create_resources_validation_fails(self, mock_load, mock_validate, mock_chassis, mock_nodes):
    resources_files = ['file.json']
    self.assertRaises(exc.ClientException, create_resources.create_resources, self.client, resources_files)
    mock_load.assert_has_calls([mock.call('file.json')])
    mock_validate.assert_called_once_with(schema_pov_invalid_json, mock.ANY)
    self.assertFalse(mock_chassis.called)
    self.assertFalse(mock_nodes.called)