import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_upload_for_the_first_time_do_a_full_upload(self):
    self.make_branch_and_working_tree()
    self.add_file('hello', b'bar')
    revid_path = self.tree.branch.get_config_stack().get('upload_revid_location')
    self.assertUpPathDoesNotExist(revid_path)
    self.do_upload()
    self.assertUpFileEqual(b'bar', 'hello')