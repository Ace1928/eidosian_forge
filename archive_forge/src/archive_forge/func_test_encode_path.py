from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_encode_path(self):
    encode = tests.GitBranchBuilder._encode_path
    self.assertEqual(encode('fµ'), b'f\xc2\xb5')
    self.assertEqual(encode('"foo'), b'"\\"foo"')
    self.assertEqual(encode('fo\no'), b'"fo\\no"')
    self.assertEqual(encode('fo\\o\nbar'), b'"fo\\\\o\\nbar"')
    self.assertEqual(encode('fo"o"\nbar'), b'"fo\\"o\\"\\nbar"')
    self.assertEqual(encode('foo\r\nbar'), b'"foo\r\\nbar"')