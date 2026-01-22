import sys
import unittest
def test_simple_level_0(self):
    LINES = ['Three blind mice', 'See how they run']
    text = '\n'.join(LINES)
    self.assertEqual(self._callFUT(text, 0), text)