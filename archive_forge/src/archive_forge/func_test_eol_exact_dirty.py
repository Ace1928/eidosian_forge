import sys
from io import BytesIO
from ... import rules, status
from ...workingtree import WorkingTree
from .. import TestSkipped
from . import TestCaseWithWorkingTree
def test_eol_exact_dirty(self):
    wt, basis = self.prepare_tree(_sample_text, eol='exact')
    self.assertContent(wt, basis, _sample_text, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=[])