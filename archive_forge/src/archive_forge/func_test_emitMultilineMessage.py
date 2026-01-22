from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_emitMultilineMessage(self):
    """
        Each line of a multiline message is emitted separately to the syslog.
        """
    self.observer.emit({'message': ('hello,\nworld',), 'isError': False, 'system': '-'})
    self.assertEqual(self.events, [(stdsyslog.LOG_INFO, '[-] hello,'), (stdsyslog.LOG_INFO, '[-] \tworld')])