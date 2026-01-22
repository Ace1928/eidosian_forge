import datetime
import unittest
from traits.util.clean_strings import clean_filename, clean_timestamp
def test_clean_filename_conversion_to_lowercase(self):
    test_string = 'ABCdefGHI123'
    with self.assertWarns(DeprecationWarning):
        safe_string = clean_filename(test_string)
    self.assertEqual(safe_string, test_string.lower())
    self.check_output(safe_string)