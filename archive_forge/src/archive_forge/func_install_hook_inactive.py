import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def install_hook_inactive(self):

    def inactive_factory(merger):
        self.hook_log.append(('inactive',))
        return None
    _mod_merge.Merger.hooks.install_named_hook('merge_file_content', inactive_factory, 'test hook (inactive)')