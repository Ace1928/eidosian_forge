import os
import unittest
import warnings
def test_skip_build(self):
    dist = self.create_dist()[1]
    cmd = bdist(dist)
    cmd.skip_build = 1
    cmd.ensure_finalized()
    dist.command_obj['bdist'] = cmd
    for name in ['bdist_dumb']:
        subcmd = cmd.get_finalized_command(name)
        if getattr(subcmd, '_unsupported', False):
            continue
        self.assertTrue(subcmd.skip_build, '%s should take --skip-build from bdist' % name)