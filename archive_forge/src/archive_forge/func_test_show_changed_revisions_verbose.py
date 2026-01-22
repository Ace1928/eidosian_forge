import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_show_changed_revisions_verbose(self):
    tree = self.make_branch_and_tree('tree_a')
    self.build_tree(['tree_a/foo'])
    tree.add('foo')
    tree.commit('bar', rev_id=b'bar-id')
    s = self.make_utf8_encoded_stringio()
    log.show_changed_revisions(tree.branch, [], [b'bar-id'], s)
    self.assertContainsRe(s.getvalue(), b'bar')
    self.assertNotContainsRe(s.getvalue(), b'foo')