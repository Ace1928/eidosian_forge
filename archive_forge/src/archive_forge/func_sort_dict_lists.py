import logging as std_logging
import os
import os.path
import random
from unittest import mock
import fixtures
from oslo_config import cfg
from oslo_db import options as db_options
from oslo_utils import strutils
import pbr.version
import testtools
from neutron_lib._i18n import _
from neutron_lib import constants
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _post_mortem_debug as post_mortem_debug
def sort_dict_lists(self, dic):
    for key, value in dic.items():
        if isinstance(value, list):
            dic[key] = sorted(value)
        elif isinstance(value, dict):
            dic[key] = self.sort_dict_lists(value)
    return dic