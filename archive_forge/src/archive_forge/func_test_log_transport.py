from breezy import transport
from breezy.tests import TestCaseWithMemoryTransport
from breezy.trace import mutter
from breezy.transport.log import TransportLogDecorator
def test_log_transport(self):
    base_transport = self.get_transport('')
    logging_transport = transport.get_transport('log+' + base_transport.base)
    mutter('where are you?')
    logging_transport.mkdir('subdir')
    log = self.get_log()
    self.assertContainsRe(log, 'mkdir subdir')
    self.assertContainsRe(log, '  --> None')
    self.assertTrue(logging_transport.has('subdir'))
    self.assertTrue(base_transport.has('subdir'))