import copy
from unittest import mock
import uuid
from oslo_config import cfg
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def test_option_dict_fails_when_group_is_none(self):
    group = 'foo'
    option = 'bar'
    self.assertRaises(cfg.NoSuchOptError, PROVIDERS.domain_config_api._option_dict, group, option)