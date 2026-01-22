from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def sort_by_dots(branch, tags):

    def sort_key(tag_and_revid):
        return tag_and_revid[0].count('.')
    tags.sort(key=sort_key)