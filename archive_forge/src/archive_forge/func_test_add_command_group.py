import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
def test_add_command_group(self):
    mgr = FakeCommandManager('test')
    mock_cmd_one = mock.Mock()
    mgr.add_command('mock', mock_cmd_one)
    cmd_mock, name, args = mgr.find_command(['mock'])
    self.assertEqual(mock_cmd_one, cmd_mock)
    cmd_one, name, args = mgr.find_command(['one'])
    self.assertEqual(FAKE_CMD_ONE, cmd_one)
    mgr.add_command_group('greek')
    cmd_alpha, name, args = mgr.find_command(['alpha'])
    self.assertEqual(FAKE_CMD_ALPHA, cmd_alpha)
    cmd_two, name, args = mgr.find_command(['two'])
    self.assertEqual(FAKE_CMD_TWO, cmd_two)