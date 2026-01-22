from twisted.python import usage
from twisted.trial import unittest
def test_subCommandParseOptionsHasParent(self):
    """
        The parseOptions method from the Options object specified for the
        given subcommand is called.
        """

    class SubOpt(usage.Options):

        def parseOptions(self, *a, **kw):
            self.sawParent = self.parent
            usage.Options.parseOptions(self, *a, **kw)

    class Opt(usage.Options):
        subCommands = [('foo', 'f', SubOpt, 'bar')]
    o = Opt()
    o.parseOptions(['foo'])
    self.assertTrue(hasattr(o.subOptions, 'sawParent'))
    self.assertEqual(o.subOptions.sawParent, o)