from twisted.python.usage import UsageError
from twisted.runner import procmontap as tap
from twisted.runner.procmon import ProcessMonitor
from twisted.trial import unittest
def test_maxRestartDelay(self) -> None:
    """
        The maxrestartdelay option is recognised as a parameter and coerced to
        float.
        """
    opt = tap.Options()
    opt.parseOptions(['--maxrestartdelay', '7.5', 'foo'])
    self.assertEqual(opt['maxrestartdelay'], 7.5)