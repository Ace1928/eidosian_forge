import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def make_tree_with_text_conflict(self):
    tb = self.make_branch_and_tree('base')
    self.build_tree_contents([('base/file', b'content in base')])
    tb.add('file')
    tb.commit('Adding file')
    t1 = tb.controldir.sprout('t1').open_workingtree()
    self.build_tree_contents([('base/file', b'content changed in base')])
    tb.commit('Changing file in base')
    self.build_tree_contents([('t1/file', b'content in t1')])
    t1.commit('Changing file in t1')
    t1.merge_from_branch(tb.branch)
    fnames = ['file.%s' % s for s in ('BASE', 'THIS', 'OTHER')]
    for fn in fnames:
        self.assertPathExists(os.path.join(t1.basedir, fn))
    return t1