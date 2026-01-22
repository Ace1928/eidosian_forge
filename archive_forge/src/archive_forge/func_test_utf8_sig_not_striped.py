import unittest
from charset_normalizer.legacy import detect
def test_utf8_sig_not_striped(self):
    r = detect('Hello World'.encode('utf-8-sig'))
    with self.subTest('Verify that UTF-8-SIG is returned when using legacy detect'):
        self.assertEqual(r['encoding'], 'UTF-8-SIG')