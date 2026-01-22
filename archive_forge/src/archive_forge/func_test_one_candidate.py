import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
def test_one_candidate(self):
    self.assertEqual(['resource provider list'], commandmanager._get_commands_by_partial_name(['r', 'p', 'l'], self.commands))
    self.assertEqual(['resource provider list'], commandmanager._get_commands_by_partial_name(['resource', 'provider', 'list'], self.commands))
    self.assertEqual(['server list'], commandmanager._get_commands_by_partial_name(['serve', 'l'], self.commands))