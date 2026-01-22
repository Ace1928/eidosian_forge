import io
import os
import sys
from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import help
from cliff.tests import base
from cliff.tests import utils
def test_list_deprecated_commands(self):
    stdout = io.StringIO()
    app = application.App('testing', '1', utils.TestCommandManager(utils.TEST_NAMESPACE), stdout=stdout)
    app.NAME = 'test'
    try:
        app.run(['--help'])
    except help.HelpExit:
        pass
    help_output = stdout.getvalue()
    self.assertIn('two words', help_output)
    self.assertIn('three word command', help_output)
    self.assertNotIn('old cmd', help_output)