from io import BytesIO
from ..errors import BinaryFile
from ..textfile import check_text_lines, check_text_path, text_file
from . import TestCase, TestCaseInTempDir
def test_check_text_lines(self):
    lines = [b'ab' * 2048]
    check_text_lines(lines)
    lines = [b'a' * 1023 + b'\x00']
    self.assertRaises(BinaryFile, check_text_lines, lines)