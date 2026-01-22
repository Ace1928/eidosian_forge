import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
def test_multiple_candidates(self):
    self.assertEqual(2, len(commandmanager._get_commands_by_partial_name(['se', 'li'], self.commands)))