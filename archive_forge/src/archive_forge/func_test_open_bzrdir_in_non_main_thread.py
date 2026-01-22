import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def test_open_bzrdir_in_non_main_thread(self):
    """Opening a bzrdir in a non-main thread should work ok.

        This makes sure that the globally-installed
        breezy.bzr.smart.request._pre_open_hook, which uses a threading.local(),
        works in a newly created thread.
        """
    bzrdir = self.make_controldir('.')
    transport = bzrdir.root_transport
    thread_result = []

    def t():
        BzrDir.open_from_transport(transport)
        thread_result.append('ok')
    thread = threading.Thread(target=t)
    thread.start()
    thread.join()
    self.assertEqual(['ok'], thread_result)