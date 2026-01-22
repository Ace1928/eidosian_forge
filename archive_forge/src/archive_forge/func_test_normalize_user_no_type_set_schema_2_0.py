import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_normalize_user_no_type_set_schema_2_0(self):
    user = {}
    self.rule_processor_schema_2_0.normalize_user(user, self.domain_mock)
    self.assertEqual(utils.UserType.EPHEMERAL, user['type'])