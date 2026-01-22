import os
from io import BytesIO
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_tree import TestCaseWithTree
from ... import revision as _mod_revision
from ... import tests, trace
from ...diff import show_diff_trees
from ...merge import Merge3Merger, Merger
from ...transform import ROOT_PARENT, resolve_conflicts
from ...tree import TreeChange, find_previous_path
from ..features import SymlinkFeature, UnicodeFilenameFeature
def get_empty_preview(self):
    repository = self.make_repository('repo')
    tree = repository.revision_tree(_mod_revision.NULL_REVISION)
    preview = tree.preview_transform()
    self.addCleanup(preview.finalize)
    return preview