from __future__ import (absolute_import, division, print_function)
import traceback
from datetime import datetime
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.okd.plugins.module_utils.openshift_ldap import (
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
@property
def k8s_group_api(self):
    if not self.__k8s_group_api:
        params = dict(kind='Group', api_version='user.openshift.io/v1', fail=True)
        self.__k8s_group_api = self.find_resource(**params)
    return self.__k8s_group_api