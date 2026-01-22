import datetime
import unittest
from traits.util.clean_strings import clean_filename, clean_timestamp
def test_clean_filename_default(self):
    test_strings = ['!!!', '', ' ', '\t/\n', '^!+']
    for test_string in test_strings:
        with self.assertWarns(DeprecationWarning):
            safe_string = clean_filename(test_string, 'default-output')
        self.check_output(safe_string)
        self.assertEqual(safe_string, 'default-output')