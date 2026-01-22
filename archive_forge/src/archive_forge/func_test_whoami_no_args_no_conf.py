from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
def test_whoami_no_args_no_conf(self):
    out = self.run_bzr('whoami')[0]
    self.assertTrue(len(out) > 0)
    self.assertEqual(1, out.count('@'))