import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def get_as_tree(self, revision_spec, tree=None):
    if tree is None:
        tree = self.tree
    spec = RevisionSpec.from_string(revision_spec)
    return spec.as_tree(tree.branch)