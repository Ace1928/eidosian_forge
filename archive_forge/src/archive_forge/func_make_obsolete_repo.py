import os
import re
import unittest
from breezy import bzr, config, controldir, errors, osutils, repository, tests
from breezy.bzr.groupcompress_repo import RepositoryFormat2a
def make_obsolete_repo(self, path):
    format = controldir.format_registry.make_controldir('testobsolete')
    tree = self.make_branch_and_tree(path, format=format)
    return tree