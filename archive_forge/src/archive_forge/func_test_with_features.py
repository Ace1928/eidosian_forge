import os
import subprocess
import sys
import breezy.branch
import breezy.bzr.branch
from ... import (branch, bzr, config, controldir, errors, help_topics, lock,
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ... import urlutils, win32utils
from ...errors import (NotBranchError, UnknownFormatError,
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from ...transport import memory, pathfilter
from ...transport.http.urllib import HttpTransport
from ...transport.nosmart import NoSmartTransportDecorator
from ...transport.readonly import ReadonlyTransportDecorator
from .. import branch as bzrbranch
from .. import (bzrdir, knitpack_repo, knitrepo, remote, workingtree_3,
from ..fullhistory import BzrBranchFormat5
def test_with_features(self):
    tree = self.make_branch_and_tree('tree', format='2a')
    tree.controldir.update_feature_flags({b'bar': b'required'})
    self.assertRaises(bzrdir.MissingFeature, bzrdir.BzrDir.open, 'tree')
    bzrdir.BzrDirMetaFormat1.register_feature(b'bar')
    self.addCleanup(bzrdir.BzrDirMetaFormat1.unregister_feature, b'bar')
    dir = bzrdir.BzrDir.open('tree')
    self.assertEqual(b'required', dir._format.features.get(b'bar'))
    tree.controldir.update_feature_flags({b'bar': None, b'nonexistant': None})
    dir = bzrdir.BzrDir.open('tree')
    self.assertEqual({}, dir._format.features)