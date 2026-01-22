import importlib
import os
from unittest import mock
from osc_lib.tests import utils as osc_lib_test_utils
import wrapt
from openstackclient import shell
def test_empty_env(self):
    os.environ = {}
    flag = ''
    kwargs = {'compute_api_version': LIB_COMPUTE_API_VERSION, 'identity_api_version': LIB_IDENTITY_API_VERSION, 'image_api_version': LIB_IMAGE_API_VERSION, 'volume_api_version': LIB_VOLUME_API_VERSION, 'network_api_version': LIB_NETWORK_API_VERSION}
    self._assert_cli(flag, kwargs)