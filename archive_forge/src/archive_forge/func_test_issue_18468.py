from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_issue_18468(self):
    self.assertTypedEqual(regex.sub('y', 'a', 'xyz'), 'xaz')
    self.assertTypedEqual(regex.sub('y', StrSubclass('a'), StrSubclass('xyz')), 'xaz')
    self.assertTypedEqual(regex.sub(b'y', b'a', b'xyz'), b'xaz')
    self.assertTypedEqual(regex.sub(b'y', BytesSubclass(b'a'), BytesSubclass(b'xyz')), b'xaz')
    self.assertTypedEqual(regex.sub(b'y', bytearray(b'a'), bytearray(b'xyz')), b'xaz')
    self.assertTypedEqual(regex.sub(b'y', memoryview(b'a'), memoryview(b'xyz')), b'xaz')
    for string in (':a:b::c', StrSubclass(':a:b::c')):
        self.assertTypedEqual(regex.split(':', string), ['', 'a', 'b', '', 'c'])
        if sys.version_info >= (3, 7, 0):
            self.assertTypedEqual(regex.split(':*', string), ['', '', 'a', '', 'b', '', 'c', ''])
            self.assertTypedEqual(regex.split('(:*)', string), ['', ':', '', '', 'a', ':', '', '', 'b', '::', '', '', 'c', '', ''])
        else:
            self.assertTypedEqual(regex.split(':*', string), ['', 'a', 'b', 'c'])
            self.assertTypedEqual(regex.split('(:*)', string), ['', ':', 'a', ':', 'b', '::', 'c'])
    for string in (b':a:b::c', BytesSubclass(b':a:b::c'), bytearray(b':a:b::c'), memoryview(b':a:b::c')):
        self.assertTypedEqual(regex.split(b':', string), [b'', b'a', b'b', b'', b'c'])
        if sys.version_info >= (3, 7, 0):
            self.assertTypedEqual(regex.split(b':*', string), [b'', b'', b'a', b'', b'b', b'', b'c', b''])
            self.assertTypedEqual(regex.split(b'(:*)', string), [b'', b':', b'', b'', b'a', b':', b'', b'', b'b', b'::', b'', b'', b'c', b'', b''])
        else:
            self.assertTypedEqual(regex.split(b':*', string), [b'', b'a', b'b', b'c'])
            self.assertTypedEqual(regex.split(b'(:*)', string), [b'', b':', b'a', b':', b'b', b'::', b'c'])
    for string in ('a:b::c:::d', StrSubclass('a:b::c:::d')):
        self.assertTypedEqual(regex.findall(':+', string), [':', '::', ':::'])
        self.assertTypedEqual(regex.findall('(:+)', string), [':', '::', ':::'])
        self.assertTypedEqual(regex.findall('(:)(:*)', string), [(':', ''), (':', ':'), (':', '::')])
    for string in (b'a:b::c:::d', BytesSubclass(b'a:b::c:::d'), bytearray(b'a:b::c:::d'), memoryview(b'a:b::c:::d')):
        self.assertTypedEqual(regex.findall(b':+', string), [b':', b'::', b':::'])
        self.assertTypedEqual(regex.findall(b'(:+)', string), [b':', b'::', b':::'])
        self.assertTypedEqual(regex.findall(b'(:)(:*)', string), [(b':', b''), (b':', b':'), (b':', b'::')])
    for string in ('a', StrSubclass('a')):
        self.assertEqual(regex.match('a', string).groups(), ())
        self.assertEqual(regex.match('(a)', string).groups(), ('a',))
        self.assertEqual(regex.match('(a)', string).group(0), 'a')
        self.assertEqual(regex.match('(a)', string).group(1), 'a')
        self.assertEqual(regex.match('(a)', string).group(1, 1), ('a', 'a'))
    for string in (b'a', BytesSubclass(b'a'), bytearray(b'a'), memoryview(b'a')):
        self.assertEqual(regex.match(b'a', string).groups(), ())
        self.assertEqual(regex.match(b'(a)', string).groups(), (b'a',))
        self.assertEqual(regex.match(b'(a)', string).group(0), b'a')
        self.assertEqual(regex.match(b'(a)', string).group(1), b'a')
        self.assertEqual(regex.match(b'(a)', string).group(1, 1), (b'a', b'a'))