import hashlib
from oslo_utils.secretutils import md5
from oslotest import base
import glance_store.driver as driver
def test_neg_bogus_kw_args(self):
    self.assertRaises(TypeError, self.fake_store.add, thrashing_algo=self.hashing_algo, image_file=self.img_file, context=self.fake_context, image_size=self.img_size, verifier=self.fake_verifier, image_id=self.img_id)