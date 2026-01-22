import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
def test_run_bzr_user_error_caught(self):
    transport_server = memory.MemoryServer()
    transport_server.start_server()
    self.addCleanup(transport_server.stop_server)
    url = transport_server.get_url()
    self.permit_url(url)
    out, err = self.run_bzr(['log', '%s/nonexistantpath' % url], retcode=3)
    self.assertEqual(out, '')
    self.assertContainsRe(err, 'brz: ERROR: Not a branch: ".*nonexistantpath/".\n')