import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def get_captured_revisions(self):
    return self.log_catcher.revisions