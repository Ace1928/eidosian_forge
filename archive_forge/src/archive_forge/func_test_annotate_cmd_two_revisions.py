from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_cmd_two_revisions(self):
    out, err = self.run_bzr('annotate hello.txt -r1..2', retcode=3)
    self.assertEqual('', out)
    self.assertEqual('brz: ERROR: brz annotate --revision takes exactly one revision identifier\n', err)