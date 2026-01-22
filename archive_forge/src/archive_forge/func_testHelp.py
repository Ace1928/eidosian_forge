from gslib.utils import system_util
from gslib.utils.user_agent_helper import GetUserAgent
import gslib.tests.testcase as testcase
import six
from six import add_move, MovedModule
from six.moves import mock
def testHelp(self):
    self.assertRegex(GetUserAgent(['help']), 'command/help')