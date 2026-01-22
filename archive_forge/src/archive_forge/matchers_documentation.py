from testtools.matchers import Matcher, Mismatch
from breezy.bzr.smart import vfs
from breezy.bzr.smart.request import request_handlers as smart_request_handlers
Ensure that none of the specified calls are HPSS calls.