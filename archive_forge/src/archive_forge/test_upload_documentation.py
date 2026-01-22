import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
Creates a branch without working tree to upload from.

        It's created from the existing self.branch_dir one which still has its
        working tree.
        