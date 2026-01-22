from breezy.tests import TestCaseWithTransport
def test_specific_command(self):
    out, err = self.run_bzr('shell-complete shell-complete')
    self.assertEqual('', err)
    self.assertEqual('"(--help -h)"{--help,-h}\n"(--quiet -q)"{--quiet,-q}\n"(--verbose -v)"{--verbose,-v}\n--usage\ncontext?\n'.splitlines(), sorted(out.splitlines()))