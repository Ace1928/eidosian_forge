from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
def test_normalize_to_crlf_binary(self):
    base_content = b'line1\r\nline2\x00'
    base_sha = 'b44504193b765f7cd79673812de8afb55b372ab2'
    base_blob = Blob()
    base_blob.set_raw_string(base_content)
    self.assertEqual(base_blob.as_raw_chunks(), [base_content])
    self.assertEqual(base_blob.sha().hexdigest(), base_sha)
    filtered_blob = normalize_blob(base_blob, convert_lf_to_crlf, binary_detection=True)
    self.assertEqual(filtered_blob.as_raw_chunks(), [base_content])
    self.assertEqual(filtered_blob.sha().hexdigest(), base_sha)