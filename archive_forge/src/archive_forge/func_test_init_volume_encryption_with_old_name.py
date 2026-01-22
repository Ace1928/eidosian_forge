import binascii
import copy
from unittest import mock
from castellan.common.objects import symmetric_key as key
from castellan.tests.unit.key_manager import fake
from os_brick.encryptors import cryptsetup
from os_brick import exception
from os_brick.tests.encryptors import test_base
@mock.patch('os_brick.executor.Executor._execute')
@mock.patch('os.path.exists', return_value=True)
def test_init_volume_encryption_with_old_name(self, mock_exists, mock_execute):
    old_dev_name = self.dev_path.split('/')[-1]
    encryptor = cryptsetup.CryptsetupEncryptor(root_helper=self.root_helper, connection_info=self.connection_info, keymgr=self.keymgr)
    self.assertFalse(encryptor.dev_name.startswith('crypt-'))
    self.assertEqual(old_dev_name, encryptor.dev_name)
    self.assertEqual(self.dev_path, encryptor.dev_path)
    self.assertEqual(self.symlink_path, encryptor.symlink_path)
    mock_exists.assert_called_once_with('/dev/mapper/%s' % old_dev_name)
    mock_execute.assert_called_once_with('cryptsetup', 'status', old_dev_name, run_as_root=True)