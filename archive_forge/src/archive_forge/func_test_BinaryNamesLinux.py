import sys
import unittest
import gyp.generator.ninja as ninja
def test_BinaryNamesLinux(self):
    writer = ninja.NinjaWriter('foo', 'wee', '.', '.', 'build.ninja', '.', 'build.ninja', 'linux')
    spec = {'target_name': 'wee'}
    self.assertTrue('.' not in writer.ComputeOutputFileName(spec, 'executable'))
    self.assertTrue(writer.ComputeOutputFileName(spec, 'shared_library').startswith('lib'))
    self.assertTrue(writer.ComputeOutputFileName(spec, 'static_library').startswith('lib'))
    self.assertTrue(writer.ComputeOutputFileName(spec, 'shared_library').endswith('.so'))
    self.assertTrue(writer.ComputeOutputFileName(spec, 'static_library').endswith('.a'))