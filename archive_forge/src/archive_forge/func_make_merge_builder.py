import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def make_merge_builder(self):
    builder = MergeBuilder(self.test_base_dir)
    self.addCleanup(builder.cleanup)
    return builder