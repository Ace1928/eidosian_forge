import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
def test_get_command_names(self):
    mock_cmd_one = mock.Mock()
    mock_cmd_one.name = 'one'
    mock_cmd_two = mock.Mock()
    mock_cmd_two.name = 'cmd two'
    mock_get_group_all = mock.Mock(return_value=[mock_cmd_one, mock_cmd_two])
    with mock.patch('stevedore.ExtensionManager', mock_get_group_all) as mock_manager:
        mgr = commandmanager.CommandManager('test')
        mock_manager.assert_called_once_with('test')
        cmds = mgr.get_command_names('test')
        self.assertEqual(['one', 'cmd two'], cmds)