import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_config_var_no_strict(self):
    self.set_config_push_strict('false')
    self.assertPushSucceeds([])