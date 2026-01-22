import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def forbidden_wrapper(f):

    def wrapped_func(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except tempest_lib_exc.CommandFailed as e:
            if re.search('HTTP 403', str(e.stderr)):
                raise tempest_lib_exc.Forbidden()
            raise
    return wrapped_func