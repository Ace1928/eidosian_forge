import gyp.generator.xcode as xcode
import unittest
import sys
def test_InheritedRemainsUnescaped(self):
    self.assertEqual(xcode.EscapeXcodeDefine('$(inherited)'), '$(inherited)')