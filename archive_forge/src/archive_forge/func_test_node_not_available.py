import logging
import subprocess
from os.path import join as pjoin
from unittest.mock import patch
from jupyterlab import commands
from .test_jupyterlab import AppHandlerTest
def test_node_not_available(self):
    with patch('jupyterlab.commands.which') as which:
        which.side_effect = ValueError('Command not found')
        logger = logging.getLogger('jupyterlab')
        config = commands._yarn_config(logger)
        which.assert_called_once_with('node')
        self.assertDictEqual(config, {'yarn config': {}, 'npm config': {}})