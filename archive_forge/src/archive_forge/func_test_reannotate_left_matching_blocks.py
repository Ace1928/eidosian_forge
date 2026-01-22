import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def test_reannotate_left_matching_blocks(self):
    """Ensure that left_matching_blocks has an impact.

        In this case, the annotation is ambiguous, so the hint isn't actually
        lying.
        """
    parent = [(b'rev1', b'a\n')]
    new_text = [b'a\n', b'a\n']
    blocks = [(0, 0, 1), (1, 2, 0)]
    self.annotateEqual([(b'rev1', b'a\n'), (b'rev2', b'a\n')], [parent], new_text, b'rev2', blocks)
    blocks = [(0, 1, 1), (1, 2, 0)]
    self.annotateEqual([(b'rev2', b'a\n'), (b'rev1', b'a\n')], [parent], new_text, b'rev2', blocks)