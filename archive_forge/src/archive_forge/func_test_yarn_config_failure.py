import logging
import subprocess
from os.path import join as pjoin
from unittest.mock import patch
from jupyterlab import commands
from .test_jupyterlab import AppHandlerTest
def test_yarn_config_failure(self):
    with patch('subprocess.check_output') as check_output:
        check_output.side_effect = subprocess.CalledProcessError(1, ['yarn', 'config', 'list'], b'', stderr=b'yarn config failed.')
        logger = logging.getLogger('jupyterlab')
        config = commands._yarn_config(logger)
        self.assertDictEqual(config, {'yarn config': {}, 'npm config': {}})