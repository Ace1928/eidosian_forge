from oslo_config import cfg
from heat.common import config
from heat.common import crypt
from heat.common import exception
from heat.tests import common
def test_decrypt_dict_invalid_key(self):
    data = {'p1': u'happy', '2': [u'a', u'little', u'blue'], '6': 7}
    encrypted_data = crypt.encrypted_dict(data, '767c3ed056cbaa3b9dfedb8c6f825bf0')
    ex = self.assertRaises(exception.InvalidEncryptionKey, crypt.decrypted_dict, encrypted_data, '767c3ed056cbaa3b9dfedb8c6f825bf1')
    self.assertEqual('Can not decrypt data with the auth_encryption_key in heat config.', str(ex))