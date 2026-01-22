from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
def test_normalize_to_lf_no_op(self):
    base_content = b'line1\nline2'
    base_sha = 'f8be7bb828880727816015d21abcbc37d033f233'
    base_blob = Blob()
    base_blob.set_raw_string(base_content)
    self.assertEqual(base_blob.as_raw_chunks(), [base_content])
    self.assertEqual(base_blob.sha().hexdigest(), base_sha)
    filtered_blob = normalize_blob(base_blob, convert_crlf_to_lf, binary_detection=False)
    self.assertEqual(filtered_blob.as_raw_chunks(), [base_content])
    self.assertEqual(filtered_blob.sha().hexdigest(), base_sha)