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
def test_mapping_validation_with_bad_local_type_user_in_assertion(self):
    mapping = mapping_fixtures.MAPPING_BAD_LOCAL_TYPE_USER_IN_ASSERTION
    self.assertRaises(exception.ValidationError, mapping_utils.validate_mapping_structure, mapping)