from twisted.python import usage
from twisted.trial import unittest
def test_multiList(self):
    """
        CompleteMultiList produces zsh shell-code that completes multiple
        comma-separated words from a fixed list of possibilities.
        """
    c = usage.CompleteMultiList('ABC')
    got = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(got, ":some-option:_values -s , 'some-option' A B C")
    c = usage.CompleteMultiList(['1', '2', '3'])
    got = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(got, ":some-option:_values -s , 'some-option' 1 2 3")
    c = usage.CompleteMultiList(['1', '2', '3'], descr='some action', repeat=True)
    got = c._shellCode('some-option', usage._ZSH)
    expected = "*:some action:_values -s , 'some action' 1 2 3"
    self.assertEqual(got, expected)