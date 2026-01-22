from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_cmd_unknown_revision(self):
    out, err = self.run_bzr('annotate hello.txt -r 10', retcode=3)
    self.assertEqual('', out)
    self.assertContainsRe(err, "Requested revision: '10' does not exist")