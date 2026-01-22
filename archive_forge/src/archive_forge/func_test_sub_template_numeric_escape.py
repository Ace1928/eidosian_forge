from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_sub_template_numeric_escape(self):
    self.assertEqual(regex.sub('x', '\\0', 'x'), '\x00')
    self.assertEqual(regex.sub('x', '\\000', 'x'), '\x00')
    self.assertEqual(regex.sub('x', '\\001', 'x'), '\x01')
    self.assertEqual(regex.sub('x', '\\008', 'x'), '\x00' + '8')
    self.assertEqual(regex.sub('x', '\\009', 'x'), '\x00' + '9')
    self.assertEqual(regex.sub('x', '\\111', 'x'), 'I')
    self.assertEqual(regex.sub('x', '\\117', 'x'), 'O')
    self.assertEqual(regex.sub('x', '\\1111', 'x'), 'I1')
    self.assertEqual(regex.sub('x', '\\1111', 'x'), 'I' + '1')
    self.assertEqual(regex.sub('x', '\\00', 'x'), '\x00')
    self.assertEqual(regex.sub('x', '\\07', 'x'), '\x07')
    self.assertEqual(regex.sub('x', '\\08', 'x'), '\x00' + '8')
    self.assertEqual(regex.sub('x', '\\09', 'x'), '\x00' + '9')
    self.assertEqual(regex.sub('x', '\\0a', 'x'), '\x00' + 'a')
    self.assertEqual(regex.sub('x', '\\400', 'x'), 'Ā')
    self.assertEqual(regex.sub('x', '\\777', 'x'), 'ǿ')
    self.assertEqual(regex.sub(b'x', b'\\400', b'x'), b'\x00')
    self.assertEqual(regex.sub(b'x', b'\\777', b'x'), b'\xff')
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\1', 'x'))
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\8', 'x'))
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\9', 'x'))
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\11', 'x'))
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\18', 'x'))
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\1a', 'x'))
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\90', 'x'))
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\99', 'x'))
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\118', 'x'))
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\11a', 'x'))
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\181', 'x'))
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.sub('x', '\\800', 'x'))
    self.assertEqual(regex.sub('(((((((((((x)))))))))))', '\\11', 'x'), 'x')
    self.assertEqual(regex.sub('((((((((((y))))))))))(.)', '\\118', 'xyz'), 'xz8')
    self.assertEqual(regex.sub('((((((((((y))))))))))(.)', '\\11a', 'xyz'), 'xza')