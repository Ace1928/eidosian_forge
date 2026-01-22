import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def test_annotate_show_ids(self):
    builder = self.create_deeply_merged_trees()
    self.assertBranchAnnotate('    rev-1 | first\n    rev-2 | second\nrev-1_1_1 | third\nrev-1_2_1 | fourth\nrev-1_3_1 | fifth\n          | sixth\n', builder.get_branch(), 'a', b'rev-6', show_ids=True, full=False)
    self.assertBranchAnnotate('    rev-1 | first\n    rev-2 | second\nrev-1_1_1 | third\nrev-1_2_1 | fourth\nrev-1_3_1 | fifth\nrev-1_3_1 | sixth\n', builder.get_branch(), 'a', b'rev-6', show_ids=True, full=True)