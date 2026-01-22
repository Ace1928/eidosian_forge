import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
Opening a bzrdir in a non-main thread should work ok.

        This makes sure that the globally-installed
        breezy.bzr.smart.request._pre_open_hook, which uses a threading.local(),
        works in a newly created thread.
        