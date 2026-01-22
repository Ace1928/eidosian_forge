import unittest
from jsbeautifier.unpackers.javascriptobfuscator import unpack, detect, smartsplit
def test_smartsplit(self):
    """Test smartsplit() function."""
    split = smartsplit

    def equals(data, result):
        return self.assertEqual(split(data), result)
    equals('', [])
    equals('"a", "b"', ['"a"', '"b"'])
    equals('"aaa","bbbb"', ['"aaa"', '"bbbb"'])
    equals('"a", "b\\""', ['"a"', '"b\\""'])