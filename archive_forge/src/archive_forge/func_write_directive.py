import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def write_directive(self, filename, source, target, revision_id, base_revision_id=None, mangle_patch=False):
    md = merge_directive.MergeDirective2.from_objects(source.repository, revision_id, 0, 0, target, base_revision_id=base_revision_id)
    if mangle_patch:
        md.patch = b'asdf\n'
    self.build_tree_contents([(filename, b''.join(md.to_lines()))])