from testtools import content
from pbr.tests import base
def test_freeze_command(self):
    """Test that freeze output is sorted in a case-insensitive manner."""
    stdout, stderr, return_code = self.run_pbr('freeze')
    self.assertEqual(0, return_code)
    pkgs = []
    for line in stdout.split('\n'):
        pkgs.append(line.split('==')[0].lower())
    pkgs_sort = sorted(pkgs[:])
    self.assertEqual(pkgs_sort, pkgs)