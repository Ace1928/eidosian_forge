import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_upload_revision(self):
    self.make_branch_and_working_tree()
    self.do_full_upload()
    self.add_file('hello', b'foo')
    self.modify_file('hello', b'bar')
    self.assertUpPathDoesNotExist('hello')
    revspec = revisionspec.RevisionSpec.from_string('2')
    self.do_upload(revision=[revspec])
    self.assertUpFileEqual(b'foo', 'hello')