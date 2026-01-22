import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_process_group_by_name_groups_different_domain(self):
    domain = {'name': 'domain1'}
    group1 = {'name': 'group1', 'domain': domain}
    group2 = {'name': 'group2', 'domain': self.domain_mock}
    group_by_domain = {'domain1': [group1]}
    result = self.rule_processor.process_group_by_name(group2, group_by_domain)
    self.assertEqual([group1, group2], list(result))
    self.assertEqual(['domain1', self.domain_id_mock], list(group_by_domain.keys()))