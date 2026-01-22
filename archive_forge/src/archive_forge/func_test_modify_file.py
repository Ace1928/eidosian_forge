import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_modify_file(self):
    self.make_branch_and_working_tree()
    self.add_file('hello', b'foo')
    self.do_full_upload()
    self.modify_file('hello', b'bar')
    self.assertUpFileEqual(b'foo', 'hello')
    self.do_upload()
    self.assertUpFileEqual(b'bar', 'hello')