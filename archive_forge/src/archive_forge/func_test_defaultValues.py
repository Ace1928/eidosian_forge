from twisted.python import usage
from twisted.trial import unittest
def test_defaultValues(self):
    """
        Default values are parsed.
        """
    argV = []
    self.usage.parseOptions(argV)
    self.assertEqual(self.usage.opts['fooint'], 392)
    self.assertIsInstance(self.usage.opts['fooint'], int)
    self.assertEqual(self.usage.opts['foofloat'], 4.23)
    self.assertIsInstance(self.usage.opts['foofloat'], float)
    self.assertIsNone(self.usage.opts['eggint'])
    self.assertIsNone(self.usage.opts['eggfloat'])