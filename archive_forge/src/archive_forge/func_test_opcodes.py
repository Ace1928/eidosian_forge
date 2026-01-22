import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_opcodes(self):

    def chk_ops(a, b, expected_codes):
        s = self._PatienceSequenceMatcher(None, a, b)
        self.assertEqual(expected_codes, s.get_opcodes())
    chk_ops('', '', [])
    chk_ops([], [], [])
    chk_ops('abc', '', [('delete', 0, 3, 0, 0)])
    chk_ops('', 'abc', [('insert', 0, 0, 0, 3)])
    chk_ops('abcd', 'abcd', [('equal', 0, 4, 0, 4)])
    chk_ops('abcd', 'abce', [('equal', 0, 3, 0, 3), ('replace', 3, 4, 3, 4)])
    chk_ops('eabc', 'abce', [('delete', 0, 1, 0, 0), ('equal', 1, 4, 0, 3), ('insert', 4, 4, 3, 4)])
    chk_ops('eabce', 'abce', [('delete', 0, 1, 0, 0), ('equal', 1, 5, 0, 4)])
    chk_ops('abcde', 'abXde', [('equal', 0, 2, 0, 2), ('replace', 2, 3, 2, 3), ('equal', 3, 5, 3, 5)])
    chk_ops('abcde', 'abXYZde', [('equal', 0, 2, 0, 2), ('replace', 2, 3, 2, 5), ('equal', 3, 5, 5, 7)])
    chk_ops('abde', 'abXYZde', [('equal', 0, 2, 0, 2), ('insert', 2, 2, 2, 5), ('equal', 2, 4, 5, 7)])
    chk_ops('abcdefghijklmnop', 'abcdefxydefghijklmnop', [('equal', 0, 6, 0, 6), ('insert', 6, 6, 6, 11), ('equal', 6, 16, 11, 21)])
    chk_ops(['hello there\n', 'world\n', 'how are you today?\n'], ['hello there\n', 'how are you today?\n'], [('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1), ('equal', 2, 3, 1, 2)])
    chk_ops('aBccDe', 'abccde', [('equal', 0, 1, 0, 1), ('replace', 1, 5, 1, 5), ('equal', 5, 6, 5, 6)])
    chk_ops('aBcDec', 'abcdec', [('equal', 0, 1, 0, 1), ('replace', 1, 2, 1, 2), ('equal', 2, 3, 2, 3), ('replace', 3, 4, 3, 4), ('equal', 4, 6, 4, 6)])
    chk_ops('aBcdEcdFg', 'abcdecdfg', [('equal', 0, 1, 0, 1), ('replace', 1, 8, 1, 8), ('equal', 8, 9, 8, 9)])
    chk_ops('aBcdEeXcdFg', 'abcdecdfg', [('equal', 0, 1, 0, 1), ('replace', 1, 2, 1, 2), ('equal', 2, 4, 2, 4), ('delete', 4, 5, 4, 4), ('equal', 5, 6, 4, 5), ('delete', 6, 7, 5, 5), ('equal', 7, 9, 5, 7), ('replace', 9, 10, 7, 8), ('equal', 10, 11, 8, 9)])