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
@mock.patch('os.listdir')
@mock.patch('os.path.isdir')
@mock.patch('keystone.cmd.doctor.ldap.configparser.ConfigParser')
def test_file_based_domain_specific_configs_formatted_correctly_raised(self, mocked_parser, mocked_isdir, mocked_listdir):
    symptom = 'symptom_LDAP_file_based_domain_specific_configs_formatted_correctly'
    self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True)
    self.config_fixture.config(group='identity', domain_configurations_from_database=False)
    mocked_isdir.return_value = True
    mocked_listdir.return_value = ['keystone.domains.conf']
    mock_instance = mock.MagicMock()
    mock_instance.read.side_effect = configparser.Error('No Section')
    mocked_parser.return_value = mock_instance
    self.assertTrue(getattr(ldap, symptom)())