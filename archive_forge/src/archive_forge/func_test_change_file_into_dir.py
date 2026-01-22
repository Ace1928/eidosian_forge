import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_change_file_into_dir(self):
    self._test_change_file_into_dir('hello')