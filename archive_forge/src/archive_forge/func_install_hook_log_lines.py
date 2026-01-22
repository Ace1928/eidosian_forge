import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def install_hook_log_lines(self):
    """Install a hook that saves the get_lines for the this, base and other
        versions of the file.
        """
    test = self

    class HookLogLines(_mod_merge.AbstractPerFileMerger):

        def merge_contents(self, merge_params):
            test.hook_log.append(('log_lines', merge_params.this_lines, merge_params.other_lines, merge_params.base_lines))
            return ('not_applicable', None)

    def hook_log_lines_factory(merger):
        return HookLogLines(merger)
    _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_log_lines_factory, 'test hook (log_lines)')