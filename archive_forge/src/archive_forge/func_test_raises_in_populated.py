import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_raises_in_populated(self):
    source = self.make_branch_and_tree('source')
    self.build_tree(['source/name'])
    source.add('name')
    source.commit('added name')
    target = self.make_branch_and_tree('target')
    self.build_tree(['target/name'])
    target.add('name')
    self.assertRaises(errors.WorkingTreeAlreadyPopulated, build_tree, source.basis_tree(), target)