import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
def test_unicode_filename(self):
    """Test when the filename are unicode."""
    self.requireFeature(features.UnicodeFilenameFeature)
    alpha, omega = ('α', 'ω')
    autf8, outf8 = (alpha.encode('utf8'), omega.encode('utf8'))
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/ren_' + alpha, b'contents\n')])
    tree.add(['ren_' + alpha], ids=[b'file-id-2'])
    self.build_tree_contents([('tree/del_' + alpha, b'contents\n')])
    tree.add(['del_' + alpha], ids=[b'file-id-3'])
    self.build_tree_contents([('tree/mod_' + alpha, b'contents\n')])
    tree.add(['mod_' + alpha], ids=[b'file-id-4'])
    tree.commit('one', rev_id=b'rev-1')
    tree.rename_one('ren_' + alpha, 'ren_' + omega)
    tree.remove('del_' + alpha)
    self.build_tree_contents([('tree/add_' + alpha, b'contents\n')])
    tree.add(['add_' + alpha], ids=[b'file-id'])
    self.build_tree_contents([('tree/mod_' + alpha, b'contents_mod\n')])
    d = get_diff_as_string(tree.basis_tree(), tree)
    self.assertContainsRe(d, b"=== renamed file 'ren_%s' => 'ren_%s'\n" % (autf8, outf8))
    self.assertContainsRe(d, b"=== added file 'add_%s'" % autf8)
    self.assertContainsRe(d, b"=== modified file 'mod_%s'" % autf8)
    self.assertContainsRe(d, b"=== removed file 'del_%s'" % autf8)