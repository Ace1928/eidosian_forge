import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_ver_matching_files(self):
    """(versioned) Search for matches or no matches only"""
    tree = self.make_branch_and_tree('.')
    contents = ['d/', 'd/aaa', 'bbb']
    self.build_tree(contents)
    tree.add(contents)
    tree.commit('Initial commit')
    streams = self.run_bzr(['grep', '--color', 'always', '-r', '1', '--files-with-matches', 'aaa'])
    self.assertEqual(streams, (''.join([FG.MAGENTA, 'd/aaa', self._rev_sep, '1', '\n']), ''))
    streams = self.run_bzr(['grep', '--color', 'always', '-r', '1', '--files-without-match', 'aaa'])
    self.assertEqual(streams, (''.join([FG.MAGENTA, 'bbb', self._rev_sep, '1', '\n']), ''))