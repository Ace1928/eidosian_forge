import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def prepare_merge_directive(self):
    self.tree1 = self.make_branch_and_tree('tree1')
    self.build_tree_contents([('tree1/file', b'a\nb\nc\nd\n')])
    self.tree1.branch.get_config_stack().set('email', 'J. Random Hacker <jrandom@example.com>')
    self.tree1.add('file')
    self.tree1.commit('foo', rev_id=b'foo-id')
    self.tree2 = self.tree1.controldir.sprout('tree2').open_workingtree()
    self.build_tree_contents([('tree1/file', b'a\nb\nc\nd\ne\n')])
    self.tree1.commit('bar', rev_id=b'bar-id')
    os.chdir('tree1')
    return (self.tree1, self.tree2)