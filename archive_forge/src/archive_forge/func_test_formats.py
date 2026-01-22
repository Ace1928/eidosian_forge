import os
import unittest
import warnings
def test_formats(self):
    dist = self.create_dist()[1]
    cmd = bdist(dist)
    cmd.formats = ['tar']
    cmd.ensure_finalized()
    self.assertEqual(cmd.formats, ['tar'])
    formats = ['bztar', 'gztar', 'rpm', 'tar', 'xztar', 'zip', 'ztar']
    found = sorted(cmd.format_command)
    self.assertEqual(found, formats)