import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def install_hook_noop(self):
    test = self

    class HookNA(_mod_merge.AbstractPerFileMerger):

        def merge_contents(self, merge_params):
            test.hook_log.append(('no-op',))
            return ('not_applicable', None)

    def hook_na_factory(merger):
        return HookNA(merger)
    _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_na_factory, 'test hook (no-op)')