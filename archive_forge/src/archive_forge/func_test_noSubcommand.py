from twisted.python import usage
from twisted.trial import unittest
def test_noSubcommand(self):
    """
        If no subcommand is specified and no default subcommand is assigned,
        a subcommand will not be implied.
        """
    o = SubCommandOptions()
    o.parseOptions(['--europian-swallow'])
    self.assertTrue(o['europian-swallow'])
    self.assertIsNone(o.subCommand)
    self.assertFalse(hasattr(o, 'subOptions'))