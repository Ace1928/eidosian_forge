import importlib
import os
from unittest import mock
from osc_lib.tests import utils as osc_lib_test_utils
import wrapt
from openstackclient import shell
def test_only_auth_url(self):
    flag = '--os-auth-url http://cloud.local:555'
    kwargs = {'token': DEFAULT_TOKEN, 'auth_url': 'http://cloud.local:555'}
    self._assert_token_auth(flag, kwargs)