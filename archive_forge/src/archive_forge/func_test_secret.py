from twisted.application import internet
from twisted.trial import unittest
from twisted.words import xmpproutertap as tap
from twisted.words.protocols.jabber import component
def test_secret(self) -> None:
    """
        The secret option is recognised as a parameter.
        """
    opt = tap.Options()
    opt.parseOptions(['--secret', 'hushhush'])
    self.assertEqual(opt['secret'], 'hushhush')