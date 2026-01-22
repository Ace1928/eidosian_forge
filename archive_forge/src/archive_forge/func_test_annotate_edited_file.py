from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_edited_file(self):
    tree = self._setup_edited_file()
    self.overrideEnv('BRZ_EMAIL', 'current@host2')
    out, err = self.run_bzr('annotate file')
    self.assertEqual('1   test@ho | foo\n2?  current | bar\n1   test@ho | gam\n', out)