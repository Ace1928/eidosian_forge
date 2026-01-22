import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def make_inner_branch(self):
    bld_inner = self.make_branch_builder('inner')
    bld_inner.start_series()
    rev1 = bld_inner.build_snapshot(None, [('add', ('', None, 'directory', '')), ('add', ('dir', None, 'directory', '')), ('add', ('dir/file1', None, 'file', b'file1 content\n')), ('add', ('file3', None, 'file', b'file3 content\n'))])
    rev4 = bld_inner.build_snapshot([rev1], [('add', ('file4', None, 'file', b'file4 content\n'))])
    rev5 = bld_inner.build_snapshot([rev4], [('rename', ('file4', 'dir/file4'))])
    rev3 = bld_inner.build_snapshot([rev1], [('modify', ('file3', b'new file3 contents\n'))])
    rev2 = bld_inner.build_snapshot([rev1], [('add', ('dir/file2', None, 'file', b'file2 content\n'))])
    bld_inner.finish_series()
    br = bld_inner.get_branch()
    return (br, [rev1, rev2, rev3, rev4, rev5])