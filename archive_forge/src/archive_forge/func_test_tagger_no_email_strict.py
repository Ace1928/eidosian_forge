import io
import time
import unittest
from fastimport import (
from :2
def test_tagger_no_email_strict(self):
    p = parser.ImportParser(io.BytesIO(b'tag refs/tags/v1.0\nfrom :xxx\ntagger Joe Wong\ndata 11\ncreate v1.0'))
    self.assertRaises(errors.BadFormat, list, p.iter_commands())