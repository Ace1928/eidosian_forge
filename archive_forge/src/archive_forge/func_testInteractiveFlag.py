from gslib.utils import system_util
from gslib.utils.user_agent_helper import GetUserAgent
import gslib.tests.testcase as testcase
import six
from six import add_move, MovedModule
from six.moves import mock
@mock.patch.object(system_util, 'IsRunningInteractively')
def testInteractiveFlag(self, mock_interactive):
    mock_interactive.return_value = True
    self.assertRegex(GetUserAgent([]), 'interactive/True')
    mock_interactive.return_value = False
    self.assertRegex(GetUserAgent([]), 'interactive/False')