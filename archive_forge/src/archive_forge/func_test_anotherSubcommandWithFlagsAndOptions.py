from twisted.python import usage
from twisted.trial import unittest
def test_anotherSubcommandWithFlagsAndOptions(self):
    """
        Flags and options of another subcommand are assigned.
        """
    o = SubCommandOptions()
    o.parseOptions(['holyquest', '--for-grail'])
    self.assertFalse(o['europian-swallow'])
    self.assertEqual(o.subCommand, 'holyquest')
    self.assertIsInstance(o.subOptions, HolyQuestOptions)
    self.assertFalse(o.subOptions['horseback'])
    self.assertTrue(o.subOptions['for-grail'])