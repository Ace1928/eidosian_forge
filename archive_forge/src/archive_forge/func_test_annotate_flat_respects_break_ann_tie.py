from .. import annotate, errors, revision, tests
from ..bzr import knit
def test_annotate_flat_respects_break_ann_tie(self):
    seen = set()

    def custom_tiebreaker(annotated_lines):
        self.assertEqual(2, len(annotated_lines))
        left = annotated_lines[0]
        self.assertEqual(2, len(left))
        self.assertEqual(b'new content\n', left[1])
        right = annotated_lines[1]
        self.assertEqual(2, len(right))
        self.assertEqual(b'new content\n', right[1])
        seen.update([left[0], right[0]])
        if left[0] < right[0]:
            return right
        else:
            return left
    self.overrideAttr(annotate, '_break_annotation_tie', custom_tiebreaker)
    self.make_many_way_common_merge_text()
    self.assertEqual([(self.fa_key, b'simple\n'), (self.fe_key, b'new content\n')], self.ann.annotate_flat(self.ff_key))
    self.assertEqual({self.fb_key, self.fc_key, self.fe_key}, seen)