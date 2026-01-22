from twisted.python import usage
from twisted.trial import unittest
def test_usernames(self):
    """
        CompleteUsernames produces zsh shell-code that completes system
        usernames.
        """
    c = usage.CompleteUsernames()
    out = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(out, ':some-option:_users')
    c = usage.CompleteUsernames(descr='some action', repeat=True)
    out = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(out, '*:some action:_users')