import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_with_revision(self):
    self.assertPushSucceeds(['-r', 'revid:added'], revid_to_push=b'added')