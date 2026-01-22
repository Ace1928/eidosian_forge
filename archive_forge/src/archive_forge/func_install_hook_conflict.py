import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def install_hook_conflict(self):
    test = self

    class HookConflict(_mod_merge.AbstractPerFileMerger):

        def merge_contents(self, merge_params):
            test.hook_log.append(('conflict',))
            if merge_params.this_path == 'name1':
                return ('conflicted', [b'text-with-conflict-markers-from-hook'])
            return ('not_applicable', None)

    def hook_conflict_factory(merger):
        return HookConflict(merger)
    _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_conflict_factory, 'test hook (delete)')