from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_emitCustomSystem(self):
    """
        L{SyslogObserver.emit} uses the value of the C{'system'} key to prefix
        the logged message.
        """
    self.observer.emit({'message': ('hello, world',), 'isError': False, 'system': 'nonDefaultSystem'})
    self.assertEqual(self.events, [(stdsyslog.LOG_INFO, '[nonDefaultSystem] hello, world')])