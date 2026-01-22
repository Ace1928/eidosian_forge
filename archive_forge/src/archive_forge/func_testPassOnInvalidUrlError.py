from gslib.utils import system_util
from gslib.utils.user_agent_helper import GetUserAgent
import gslib.tests.testcase as testcase
import six
from six import add_move, MovedModule
from six.moves import mock
@mock.patch.object(system_util, 'InvokedViaCloudSdk')
def testPassOnInvalidUrlError(self, mock_invoked):
    mock_invoked.return_value = False
    self.assertRegex(GetUserAgent(['cp', '-r', '-Z', 'bad://src', 's3://dst']), 'command/cp$')