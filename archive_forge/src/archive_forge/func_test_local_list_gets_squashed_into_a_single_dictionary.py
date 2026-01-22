import flask
import uuid
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.auth.plugins import mapped
import keystone.conf
from keystone import exception
from keystone.federation import utils as mapping_utils
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from unittest import mock
def test_local_list_gets_squashed_into_a_single_dictionary(self):
    expected = {'user': {'name': 'a_user', 'type': 'ephemeral'}, 'projects': [], 'group_ids': ['d34db33f'], 'group_names': []}
    mapped_split = self.process(self.mapping_split['rules'])
    mapped_combined = self.process(self.mapping_combined['rules'])
    self.assertEqual(expected, mapped_split)
    self.assertEqual(mapped_split, mapped_combined)