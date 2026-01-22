from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_named_lists(self):
    options = ['one', 'two', 'three']
    self.assertEqual(regex.match('333\\L<bar>444', '333one444', bar=options).group(), '333one444')
    self.assertEqual(regex.match('(?i)333\\L<bar>444', '333TWO444', bar=options).group(), '333TWO444')
    self.assertEqual(regex.match('333\\L<bar>444', '333four444', bar=options), None)
    options = [b'one', b'two', b'three']
    self.assertEqual(regex.match(b'333\\L<bar>444', b'333one444', bar=options).group(), b'333one444')
    self.assertEqual(regex.match(b'(?i)333\\L<bar>444', b'333TWO444', bar=options).group(), b'333TWO444')
    self.assertEqual(regex.match(b'333\\L<bar>444', b'333four444', bar=options), None)
    self.assertEqual(repr(type(regex.compile('3\\L<bar>4\\L<bar>+5', bar=['one', 'two', 'three']))), self.PATTERN_CLASS)
    self.assertEqual(regex.findall('^\\L<options>', 'solid QWERT', options=set(['good', 'brilliant', '+s\\ol[i}d'])), [])
    self.assertEqual(regex.findall('^\\L<options>', '+solid QWERT', options=set(['good', 'brilliant', '+solid'])), ['+solid'])
    options = ['STRASSE']
    self.assertEqual(regex.match('(?fi)\\L<words>', 'straße', words=options).span(), (0, 6))
    options = ['STRASSE', 'stress']
    self.assertEqual(regex.match('(?fi)\\L<words>', 'straße', words=options).span(), (0, 6))
    options = ['straße']
    self.assertEqual(regex.match('(?fi)\\L<words>', 'STRASSE', words=options).span(), (0, 7))
    options = ['kit']
    self.assertEqual(regex.search('(?i)\\L<words>', 'SKITS', words=options).span(), (1, 4))
    self.assertEqual(regex.search('(?i)\\L<words>', 'SKİTS', words=options).span(), (1, 4))
    self.assertEqual(regex.search('(?fi)\\b(\\w+) +\\1\\b', ' straße STRASSE ').span(), (1, 15))
    self.assertEqual(regex.search('(?fi)\\b(\\w+) +\\1\\b', ' STRASSE straße ').span(), (1, 15))
    self.assertEqual(regex.search('^\\L<options>$', '', options=[]).span(), (0, 0))