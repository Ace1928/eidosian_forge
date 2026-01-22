import hashlib
from oslo_utils.secretutils import md5
from oslotest import base
import glance_store.driver as driver
def test_new_style_4_args(self):
    x = self.fake_store.add(self.img_id, self.img_file, self.img_size, self.hashing_algo)
    self.assertEqual(tuple, type(x))
    self.assertEqual(5, len(x))
    self.assertIn(self.img_id, x[0])
    self.assertEqual(self.img_size, x[1])
    self.assertEqual(self.img_checksum, x[2])
    self.assertEqual(self.img_sha256, x[3])
    self.assertIsInstance(x[4], dict)
    self.assertIsNone(x[4]['context_obj'])
    self.assertIsNone(x[4]['verifier_obj'])