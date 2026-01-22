import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
def test_load_directory_after_file_is_emptied(self):

    def dict_rules(enforcer_rules):
        """Converts enforcer rules to dictionary.

            :param enforcer_rules: enforcer rules represented as a class Rules
            :return: enforcer rules represented as a dictionary
            """
        return jsonutils.loads(str(enforcer_rules))
    self.assertEqual(self.enforcer.rules, {})
    self.enforcer.load_rules()
    main_policy_file_rules = jsonutils.loads(POLICY_JSON_CONTENTS)
    self.assertEqual(main_policy_file_rules, dict_rules(self.enforcer.rules))
    folder_policy_file = os.path.join('policy.d', 'a.conf')
    self.create_config_file(folder_policy_file, POLICY_A_CONTENTS)
    self.enforcer.load_rules()
    expected_rules = main_policy_file_rules.copy()
    expected_rules.update(jsonutils.loads(POLICY_A_CONTENTS))
    self.assertEqual(expected_rules, dict_rules(self.enforcer.rules))
    self.create_config_file(folder_policy_file, '{}')
    absolute_folder_policy_file_path = self.get_config_file_fullname(folder_policy_file)
    stinfo = os.stat(absolute_folder_policy_file_path)
    os.utime(absolute_folder_policy_file_path, (stinfo.st_atime + 42, stinfo.st_mtime + 42))
    self.enforcer.load_rules()
    self.assertEqual(main_policy_file_rules, dict_rules(self.enforcer.rules))