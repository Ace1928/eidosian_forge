import copy
from testtools import matchers
from keystone.common import json_home
from keystone.tests import unit
def test_build_v3_extension_parameter_relation(self):
    extension_name = self.getUniqueString()
    extension_version = self.getUniqueString()
    parameter_name = self.getUniqueString()
    relation = json_home.build_v3_extension_parameter_relation(extension_name, extension_version, parameter_name)
    exp_relation = 'https://docs.openstack.org/api/openstack-identity/3/ext/%s/%s/param/%s' % (extension_name, extension_version, parameter_name)
    self.assertThat(relation, matchers.Equals(exp_relation))