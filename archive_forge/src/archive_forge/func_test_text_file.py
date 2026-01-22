from io import BytesIO
from ..errors import BinaryFile
from ..textfile import check_text_lines, check_text_path, text_file
from . import TestCase, TestCaseInTempDir
def test_text_file(self):
    with open('boo', 'wb') as f:
        f.write(b'ab' * 2048)
    check_text_path('boo')
    with open('boo', 'wb') as f:
        f.write(b'a' * 1023 + b'\x00')
    self.assertRaises(BinaryFile, check_text_path, 'boo')