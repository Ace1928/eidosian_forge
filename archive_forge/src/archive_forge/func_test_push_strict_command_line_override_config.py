import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_strict_command_line_override_config(self):
    self.set_config_push_strict('oFF')
    self.assertPushFails(['--strict'])
    self.assertPushSucceeds([])