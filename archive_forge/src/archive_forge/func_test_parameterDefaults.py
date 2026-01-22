from twisted.python.usage import UsageError
from twisted.runner import procmontap as tap
from twisted.runner.procmon import ProcessMonitor
from twisted.trial import unittest
def test_parameterDefaults(self) -> None:
    """
        The parameters all have default values
        """
    opt = tap.Options()
    opt.parseOptions(['foo'])
    self.assertEqual(opt['threshold'], 1)
    self.assertEqual(opt['killtime'], 5)
    self.assertEqual(opt['minrestartdelay'], 1)
    self.assertEqual(opt['maxrestartdelay'], 3600)