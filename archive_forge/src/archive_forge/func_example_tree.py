from breezy import controldir
from breezy.tests import TestCaseWithTransport
def example_tree(self, path='.', format=None):
    tree = self.make_branch_and_tree(path, format=format)
    self.build_tree_contents([(path + '/hello', b'foo')])
    tree.add('hello')
    tree.commit(message='setup')
    self.build_tree_contents([(path + '/goodbye', b'baz')])
    tree.add('goodbye')
    tree.commit(message='setup')
    return tree