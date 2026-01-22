import unittest
from charset_normalizer.legacy import detect
def test_detect_dict_keys(self):
    r = detect((u'\ufeff' + '我没有埋怨，磋砣的只是一些时间。').encode('gb18030'))
    with self.subTest('encoding key present'):
        self.assertIn('encoding', r.keys())
    with self.subTest('language key present'):
        self.assertIn('language', r.keys())
    with self.subTest('confidence key present'):
        self.assertIn('confidence', r.keys())