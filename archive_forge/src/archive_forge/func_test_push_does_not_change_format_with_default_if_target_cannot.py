import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_does_not_change_format_with_default_if_target_cannot(self):
    self.make_branch('stack_on', format='pack-0.92')
    self.make_controldir('.').get_config().set_default_stack_on('stack_on')
    self.make_branch('from', format='pack-0.92')
    out, err = self.run_bzr('push -d from to')
    b = branch.Branch.open('to')
    self.assertRaises(branch.UnstackableBranchFormat, b.get_stacked_on_url)