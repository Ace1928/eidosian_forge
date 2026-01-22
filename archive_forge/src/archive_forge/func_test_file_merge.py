import codecs
import errno
import os
import sys
import time
from io import BytesIO, StringIO
import fastbencode as bencode
from .. import filters, osutils
from .. import revision as _mod_revision
from .. import rules, tests, trace, transform, urlutils
from ..bzr import generate_ids
from ..bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ..controldir import ControlDir
from ..diff import show_diff_trees
from ..errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ..merge import Merge3Merger, Merger
from ..mutabletree import MutableTree
from ..osutils import file_kind, pathjoin
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..transport import FileExists
from . import TestCaseInTempDir, TestSkipped, features
from .features import HardlinkFeature, SymlinkFeature
def test_file_merge(self):
    self.requireFeature(SymlinkFeature(self.test_dir))
    root_id = generate_ids.gen_root_id()
    base = TransformGroup('BASE', root_id)
    this = TransformGroup('THIS', root_id)
    other = TransformGroup('OTHER', root_id)
    for tg in (this, base, other):
        tg.tt.new_directory('a', tg.root, b'a')
        tg.tt.new_symlink('b', tg.root, 'b', b'b')
        tg.tt.new_file('c', tg.root, [b'c'], b'c')
        tg.tt.new_symlink('d', tg.root, tg.name, b'd')
    targets = ((base, 'base-e', 'base-f', None, None), (this, 'other-e', 'this-f', 'other-g', 'this-h'), (other, 'other-e', None, 'other-g', 'other-h'))
    for tg, e_target, f_target, g_target, h_target in targets:
        for link, target in (('e', e_target), ('f', f_target), ('g', g_target), ('h', h_target)):
            if target is not None:
                tg.tt.new_symlink(link, tg.root, target, link.encode('ascii'))
    for tg in (this, base, other):
        tg.tt.apply()
    Merge3Merger(this.wt, this.wt, base.wt, other.wt)
    self.assertIs(os.path.isdir(this.wt.abspath('a')), True)
    self.assertIs(os.path.islink(this.wt.abspath('b')), True)
    self.assertIs(os.path.isfile(this.wt.abspath('c')), True)
    for suffix in ('THIS', 'BASE', 'OTHER'):
        self.assertEqual(os.readlink(this.wt.abspath('d.' + suffix)), suffix)
    self.assertIs(os.path.lexists(this.wt.abspath('d')), False)
    self.assertEqual(this.wt.id2path(b'd'), 'd.OTHER')
    self.assertEqual(this.wt.id2path(b'f'), 'f.THIS')
    self.assertEqual(os.readlink(this.wt.abspath('e')), 'other-e')
    self.assertIs(os.path.lexists(this.wt.abspath('e.THIS')), False)
    self.assertIs(os.path.lexists(this.wt.abspath('e.OTHER')), False)
    self.assertIs(os.path.lexists(this.wt.abspath('e.BASE')), False)
    self.assertIs(os.path.lexists(this.wt.abspath('g')), True)
    self.assertIs(os.path.lexists(this.wt.abspath('g.BASE')), False)
    self.assertIs(os.path.lexists(this.wt.abspath('h')), False)
    self.assertIs(os.path.lexists(this.wt.abspath('h.BASE')), False)
    self.assertIs(os.path.lexists(this.wt.abspath('h.THIS')), True)
    self.assertIs(os.path.lexists(this.wt.abspath('h.OTHER')), True)