from breezy import config, tests
from breezy.tests import features
def test_unicode_alias(self):
    """Unicode aliases should work (Bug #529930)"""
    self.requireFeature(features.UnicodeFilenameFeature)
    file_name = 'foo¶'
    tree = self.make_branch_and_tree('.')
    self.build_tree([file_name])
    tree.add(file_name)
    tree.commit('added')
    config.GlobalConfig.from_string('[ALIASES]\nust=st {}\n'.format(file_name), save=True)
    out, err = self.run_bzr('ust')
    self.assertEqual(err, '')
    self.assertEqual(out, '')