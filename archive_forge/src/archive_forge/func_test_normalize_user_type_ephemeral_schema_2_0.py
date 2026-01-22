import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_normalize_user_type_ephemeral_schema_2_0(self):
    user = {'type': utils.UserType.EPHEMERAL}
    self.rule_processor_schema_2_0.normalize_user(user, self.domain_mock)
    self.assertEqual(utils.UserType.EPHEMERAL, user['type'])