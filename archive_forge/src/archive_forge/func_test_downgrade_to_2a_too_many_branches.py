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
def test_downgrade_to_2a_too_many_branches(self):
    tree = self.make_branch_and_tree('.', format='development-colo')
    tree.controldir.create_branch(name='another-colocated-branch')
    converter = tree.controldir._format.get_converter(bzrdir.BzrDirMetaFormat1())
    result = converter.convert(tree.controldir, bzrdir.BzrDirMetaFormat1())
    self.assertIsInstance(result._format, bzrdir.BzrDirMetaFormat1)