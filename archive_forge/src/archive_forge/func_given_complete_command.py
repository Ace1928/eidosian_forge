from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
def given_complete_command(self):
    cmd_mgr = commandmanager.CommandManager('cliff.tests')
    app = application.App('testing', '1', cmd_mgr, stdout=FakeStdout())
    sot = complete.CompleteCommand(app, mock.Mock())
    cmd_mgr.add_command('complete', complete.CompleteCommand)
    return (sot, app, cmd_mgr)