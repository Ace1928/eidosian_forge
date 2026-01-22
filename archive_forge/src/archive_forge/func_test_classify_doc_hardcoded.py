from ...tests import TestCase
from .classify import classify_filename
def test_classify_doc_hardcoded(self):
    self.assertEqual('documentation', classify_filename('README'))