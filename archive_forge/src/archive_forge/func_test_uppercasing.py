from twisted.python import usage
from twisted.trial import unittest
def test_uppercasing(self):
    """
        Error output case adjustment does not mangle options
        """
    opt = WellBehaved()
    e = self.assertRaises(usage.UsageError, opt.parseOptions, ['-Z'])
    self.assertEqual(str(e), 'option -Z not recognized')