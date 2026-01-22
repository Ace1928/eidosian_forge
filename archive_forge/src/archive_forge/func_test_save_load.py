from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_save_load(self):
    vf = multiparent.MultiVersionedFile('foop')
    vf.add_version(b'a\nb\nc\nd'.splitlines(True), b'a', [])
    vf.add_version(b'a\ne\nd\n'.splitlines(True), b'b', [b'a'])
    vf.save()
    newvf = multiparent.MultiVersionedFile('foop')
    newvf.load()
    self.assertEqual(b'a\nb\nc\nd', b''.join(newvf.get_line_list([b'a'])[0]))
    self.assertEqual(b'a\ne\nd\n', b''.join(newvf.get_line_list([b'b'])[0]))