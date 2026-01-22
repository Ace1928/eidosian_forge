from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
def test_validate_show_nested_false(self):
    result = self.manager.validate(**{'show_nested': False})
    self.assertEqual(self.mock_response.json.return_value, result)
    self.mock_client.post.assert_called_once_with('/validate')