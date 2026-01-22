from breezy import merge_directive
from breezy.bzr import chk_map
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def create_merge_directive(self, source_branch, submit_url):
    return merge_directive.MergeDirective2.from_objects(source_branch.repository, source_branch.last_revision(), time=1247775710, timezone=0, target_branch=submit_url)