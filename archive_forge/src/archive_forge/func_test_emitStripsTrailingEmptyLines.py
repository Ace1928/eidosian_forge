from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_emitStripsTrailingEmptyLines(self):
    """
        Trailing empty lines of a multiline message are omitted from the
        messages sent to the syslog.
        """
    self.observer.emit({'message': ('hello,\nworld\n\n',), 'isError': False, 'system': '-'})
    self.assertEqual(self.events, [(stdsyslog.LOG_INFO, '[-] hello,'), (stdsyslog.LOG_INFO, '[-] \tworld')])