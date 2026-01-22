import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_copy_of_modified_file_to_new_dir(self):
    handler, branch = self.get_handler()
    src_path = b'd1/a'
    dest_path = b'd2/a'
    handler.process(self.file_command_iter(src_path, dest_path))
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(src_path,)], expected_added=[(b'd2',), (dest_path,)])
    self.assertContent(branch, revtree1, src_path, b'aaa')
    self.assertContent(branch, revtree2, src_path, b'bbb')
    self.assertContent(branch, revtree2, dest_path, b'bbb')