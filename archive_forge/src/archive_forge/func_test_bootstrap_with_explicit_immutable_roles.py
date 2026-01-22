import copy
import datetime
import logging
import os
from unittest import mock
import uuid
import argparse
import configparser
import fixtures
import freezegun
import http.client
import oslo_config.fixture
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_upgradecheck import upgradecheck
from testtools import matchers
from keystone.cmd import cli
from keystone.cmd.doctor import caching
from keystone.cmd.doctor import credential
from keystone.cmd.doctor import database as doc_database
from keystone.cmd.doctor import debug
from keystone.cmd.doctor import federation
from keystone.cmd.doctor import ldap
from keystone.cmd.doctor import security_compliance
from keystone.cmd.doctor import tokens
from keystone.cmd.doctor import tokens_fernet
from keystone.cmd import status
from keystone.common import provider_api
from keystone.common.sql import upgrades
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping as identity_mapping
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.ksfixtures import policy
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import mapping_fixtures
def test_bootstrap_with_explicit_immutable_roles(self):
    CONF(args=['bootstrap', '--bootstrap-password', uuid.uuid4().hex, '--immutable-roles'], project='keystone')
    self._do_test_bootstrap(self.bootstrap)
    admin_role = PROVIDERS.role_api.get_role(self.bootstrap.role_id)
    reader_role = PROVIDERS.role_api.get_role(self.bootstrap.reader_role_id)
    member_role = PROVIDERS.role_api.get_role(self.bootstrap.member_role_id)
    self.assertTrue(admin_role['options']['immutable'])
    self.assertTrue(member_role['options']['immutable'])
    self.assertTrue(reader_role['options']['immutable'])