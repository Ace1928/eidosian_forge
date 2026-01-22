import os.path
import shutil
import fixtures
import oslo_middleware
from glance.api.middleware import context
from glance.common import config
from glance.tests import utils as test_utils
def test_load_paste_app_paste_config_not_found(self):
    expected_middleware = context.UnauthenticatedContextMiddleware
    self.assertRaises(RuntimeError, self._do_test_load_paste_app, expected_middleware, make_paste_file=False)