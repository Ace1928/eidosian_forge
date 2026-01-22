import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def test_can_access_this_other_and_base_versions(self):
    """The hook function can call params.merger.get_lines to access the
        THIS/OTHER/BASE versions of the file.
        """
    self.install_hook_log_lines()
    builder = self.make_merge_builder()
    name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
    builder.change_contents(name1, this=b'text2', other=b'text3')
    conflicts = builder.merge(self.merge_type)
    self.assertEqual([('log_lines', [b'text2'], [b'text3'], [b'text1'])], self.hook_log)