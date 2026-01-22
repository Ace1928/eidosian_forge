from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def check_regexes(self, repo):
    if repo.supports_rich_root():
        count = 4
    else:
        count = 2
    return ('%d inconsistent parents' % count, '\\* a-file-id version broken-revision-1-2 has parents \\(parent-2, parent-1\\) but should have \\(parent-1, parent-2\\)', '\\* a-file-id version broken-revision-2-1 has parents \\(parent-1, parent-2\\) but should have \\(parent-2, parent-1\\)')