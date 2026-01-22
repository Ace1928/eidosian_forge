from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_emitCustomPriorityOverridesError(self):
    """
        L{SyslogObserver.emit} uses the value of the C{'syslogPriority'} key if
        it is specified even if the event dictionary represents an error.
        """
    self.observer.emit({'message': ('hello, world',), 'isError': True, 'system': '-', 'syslogPriority': stdsyslog.LOG_NOTICE, 'failure': Failure(Exception('bar'))})
    self.assertEqual(self.events, [(stdsyslog.LOG_NOTICE, '[-] hello, world')])