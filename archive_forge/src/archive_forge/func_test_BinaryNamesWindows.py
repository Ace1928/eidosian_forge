import sys
import unittest
import gyp.generator.ninja as ninja
def test_BinaryNamesWindows(self):
    if sys.platform.startswith('win'):
        writer = ninja.NinjaWriter('foo', 'wee', '.', '.', 'build.ninja', '.', 'build.ninja', 'win')
        spec = {'target_name': 'wee'}
        self.assertTrue(writer.ComputeOutputFileName(spec, 'executable').endswith('.exe'))
        self.assertTrue(writer.ComputeOutputFileName(spec, 'shared_library').endswith('.dll'))
        self.assertTrue(writer.ComputeOutputFileName(spec, 'static_library').endswith('.lib'))