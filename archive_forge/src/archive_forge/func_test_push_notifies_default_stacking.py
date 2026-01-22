import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_notifies_default_stacking(self):
    self.make_branch('stack_on', format='1.6')
    self.make_controldir('.').get_config().set_default_stack_on('stack_on')
    self.make_branch('from', format='1.6')
    out, err = self.run_bzr('push -d from to')
    self.assertContainsRe(err, 'Using default stacking branch stack_on at .*')