import os.path
import shutil
import fixtures
import oslo_middleware
from glance.api.middleware import context
from glance.common import config
from glance.tests import utils as test_utils
def test_get_path_non_exist(self):
    self.assertRaises(RuntimeError, config._get_deployment_config_file)