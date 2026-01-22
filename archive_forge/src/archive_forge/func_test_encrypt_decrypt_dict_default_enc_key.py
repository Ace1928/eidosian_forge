from oslo_config import cfg
from heat.common import config
from heat.common import crypt
from heat.common import exception
from heat.tests import common
def test_encrypt_decrypt_dict_default_enc_key(self):
    self._test_encrypt_decrypt_dict()