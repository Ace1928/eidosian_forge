from unittest import mock
from mistralclient.api.v2 import actions
from mistralclient.commands.v2 import actions as action_cmd
from mistralclient.commands.v2 import base as cmd_base
from mistralclient.tests.unit import base
@mock.patch('argparse.open', create=True)
def test_create_long_input(self, mock_open):
    action_long_input_dict = ACTION_DICT.copy()
    long_input = ', '.join(['var%s' % i for i in range(10)])
    action_long_input_dict['input'] = long_input
    workflow_long_input = actions.Action(mock.Mock(), action_long_input_dict)
    self.client.actions.create.return_value = [workflow_long_input]
    result = self.call(action_cmd.Create, app_args=['1.txt'])
    self.assertEqual([('1234-4567-7894-7895', 'a', True, cmd_base.cut(long_input), 'My cool action', 'test', '1', '1')], result[1])