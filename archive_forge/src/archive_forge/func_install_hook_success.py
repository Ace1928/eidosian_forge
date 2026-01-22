import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def install_hook_success(self):
    test = self

    class HookSuccess(_mod_merge.AbstractPerFileMerger):

        def merge_contents(self, merge_params):
            test.hook_log.append(('success',))
            if merge_params.this_path == 'name1':
                return ('success', [b'text-merged-by-hook'])
            return ('not_applicable', None)

    def hook_success_factory(merger):
        return HookSuccess(merger)
    _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_success_factory, 'test hook (success)')