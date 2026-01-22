from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_cmd_show_ids(self):
    out, err = self.run_bzr('annotate hello.txt --show-ids')
    max_len = max([len(self.revision_id_1), len(self.revision_id_3), len(self.revision_id_4)])
    self.assertEqual('', err)
    self.assertEqualDiff('%*s | my helicopter\n%*s | your helicopter\n%*s | all of\n%*s | our helicopters\n' % (max_len, self.revision_id_1.decode('utf-8'), max_len, self.revision_id_3.decode('utf-8'), max_len, self.revision_id_4.decode('utf-8'), max_len, ''), out)