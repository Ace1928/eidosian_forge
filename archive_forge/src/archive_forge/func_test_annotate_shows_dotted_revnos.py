import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def test_annotate_shows_dotted_revnos(self):
    builder = self.create_merged_trees()
    self.assertBranchAnnotate('1     joe@foo | first\n2     joe@foo | second\n1.1.1 barry@f | third\n', builder.get_branch(), 'a', b'rev-3')