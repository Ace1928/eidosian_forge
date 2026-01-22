import binascii
import copy
from unittest import mock
from castellan.common.objects import symmetric_key as key
from castellan.tests.unit.key_manager import fake
from oslo_concurrency import processutils as putils
from os_brick.encryptors import luks
from os_brick import exception
from os_brick.tests.encryptors import test_base
@mock.patch('os_brick.executor.Executor._execute')
def test__format_volume(self, mock_execute):
    self.encryptor._format_volume('passphrase')
    mock_execute.assert_has_calls([mock.call('cryptsetup', '--batch-mode', 'luksFormat', '--type', 'luks2', '--key-file=-', self.dev_path, process_input='passphrase', root_helper=self.root_helper, run_as_root=True, check_exit_code=True, attempts=3)])