from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
def test_complete_command_parser(self):
    sot = complete.CompleteCommand(mock.Mock(), mock.Mock())
    parser = sot.get_parser('nothing')
    self.assertEqual('nothing', parser.prog)
    self.assertEqual('print bash completion command\n    ', parser.description)