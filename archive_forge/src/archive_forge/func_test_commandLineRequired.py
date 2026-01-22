from twisted.python.usage import UsageError
from twisted.runner import procmontap as tap
from twisted.runner.procmon import ProcessMonitor
from twisted.trial import unittest
def test_commandLineRequired(self) -> None:
    """
        The command line arguments must be provided.
        """
    opt = tap.Options()
    self.assertRaises(UsageError, opt.parseOptions, [])