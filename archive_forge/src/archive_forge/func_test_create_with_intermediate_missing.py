import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_create_with_intermediate_missing(self):
    ignore_path = bedding.user_ignore_config_path()
    self.assertPathDoesNotExist(ignore_path)
    os.mkdir('empty-home')
    config_path = os.path.join(self.test_dir, 'empty-home', 'foo', '.config')
    self.overrideEnv('BRZ_HOME', config_path)
    self.assertPathDoesNotExist(config_path)
    user_ignores = ignores.get_user_ignores()
    self.assertEqual(set(ignores.USER_DEFAULTS), user_ignores)
    ignore_path = bedding.user_ignore_config_path()
    self.assertPathDoesNotExist(ignore_path)