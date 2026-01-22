from unittest import mock
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib.tests import fakes as test_fakes
from osc_lib.tests import utils as test_utils
def test_command_has_logger(self):
    cmd = FakeCommand(mock.Mock(), mock.Mock())
    self.assertTrue(hasattr(cmd, 'log'))
    self.assertEqual('osc_lib.tests.command.test_command.FakeCommand', cmd.log.name)