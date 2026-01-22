import io
import os
import sys
from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import help
from cliff.tests import base
from cliff.tests import utils
@mock.patch.object(commandmanager.EntryPointWrapper, 'load', side_effect=Exception('Could not load EntryPoint'))
def test_show_help_print_exc_with_ep_load_fail(self, mock_load):
    stdout = io.StringIO()
    app = application.App('testing', '1', utils.TestCommandManager(utils.TEST_NAMESPACE), stdout=stdout)
    app.NAME = 'test'
    app.options = mock.Mock()
    app.options.debug = True
    help_cmd = help.HelpCommand(app, mock.Mock())
    parser = help_cmd.get_parser('test')
    parsed_args = parser.parse_args([])
    try:
        help_cmd.run(parsed_args)
    except help.HelpExit:
        pass
    help_output = stdout.getvalue()
    self.assertIn('Commands:', help_output)
    self.assertIn('Could not load', help_output)
    self.assertIn('Exception: Could not load EntryPoint', help_output)