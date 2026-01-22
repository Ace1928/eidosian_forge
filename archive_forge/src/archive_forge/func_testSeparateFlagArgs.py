from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testSeparateFlagArgs(self):
    self.assertEqual(parser.SeparateFlagArgs([]), ([], []))
    self.assertEqual(parser.SeparateFlagArgs(['a', 'b']), (['a', 'b'], []))
    self.assertEqual(parser.SeparateFlagArgs(['a', 'b', '--']), (['a', 'b'], []))
    self.assertEqual(parser.SeparateFlagArgs(['a', 'b', '--', 'c']), (['a', 'b'], ['c']))
    self.assertEqual(parser.SeparateFlagArgs(['--']), ([], []))
    self.assertEqual(parser.SeparateFlagArgs(['--', 'c', 'd']), ([], ['c', 'd']))
    self.assertEqual(parser.SeparateFlagArgs(['a', 'b', '--', 'c', 'd']), (['a', 'b'], ['c', 'd']))
    self.assertEqual(parser.SeparateFlagArgs(['a', 'b', '--', 'c', 'd', '--']), (['a', 'b', '--', 'c', 'd'], []))
    self.assertEqual(parser.SeparateFlagArgs(['a', 'b', '--', 'c', '--', 'd']), (['a', 'b', '--', 'c'], ['d']))