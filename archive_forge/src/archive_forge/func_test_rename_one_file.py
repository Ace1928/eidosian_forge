import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_rename_one_file(self):
    self._test_rename_one_file('hello', 'goodbye')