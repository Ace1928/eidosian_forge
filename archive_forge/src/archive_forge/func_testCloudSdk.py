from gslib.utils import system_util
from gslib.utils.user_agent_helper import GetUserAgent
import gslib.tests.testcase as testcase
import six
from six import add_move, MovedModule
from six.moves import mock
@mock.patch.object(system_util, 'CloudSdkVersion')
@mock.patch.object(system_util, 'InvokedViaCloudSdk')
def testCloudSdk(self, mock_invoked, mock_version):
    mock_invoked.return_value = True
    mock_version.return_value = '500.1'
    self.assertRegex(GetUserAgent(['help']), 'google-cloud-sdk/500.1$')
    mock_invoked.return_value = False
    mock_version.return_value = '500.1'
    self.assertRegex(GetUserAgent(['help']), 'command/help$')