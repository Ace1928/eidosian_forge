from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_emitCustomFacility(self):
    """
        L{SyslogObserver.emit} uses the value of the C{'syslogPriority'} as the
        syslog priority, if that key is present in the event dictionary.
        """
    self.observer.emit({'message': ('hello, world',), 'isError': False, 'system': '-', 'syslogFacility': stdsyslog.LOG_CRON})
    self.assertEqual(self.events, [(stdsyslog.LOG_INFO | stdsyslog.LOG_CRON, '[-] hello, world')])