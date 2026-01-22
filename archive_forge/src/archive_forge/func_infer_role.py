from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils, role_utils
def infer_role(params):
    if params['role']:
        return ('Role', params['role'])
    return ('ClusterRole', params['cluster_role'])