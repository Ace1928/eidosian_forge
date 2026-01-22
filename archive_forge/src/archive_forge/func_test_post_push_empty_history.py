from io import BytesIO
from testtools.matchers import Equals, MatchesAny
from ... import branch, check, controldir, errors, push, tests
from ...branch import BindingUnsupported, Branch
from ...bzr import branch as bzrbranch
from ...bzr import vf_repository
from ...bzr.smart.repository import SmartServerRepositoryGetParentMap
from ...controldir import ControlDir
from ...revision import NULL_REVISION
from .. import test_server
from . import TestCaseWithInterBranch
def test_post_push_empty_history(self):
    target = self.make_to_branch('target')
    source = self.make_from_branch('source')
    Branch.hooks.install_named_hook('post_push', self.capture_post_push_hook, None)
    source.push(target)
    self.assertEqual([('post_push', source, None, target.base, 0, NULL_REVISION, 0, NULL_REVISION, True, None, True)], self.hook_calls)