from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def retrieve_distinguished_name(distinguished_name, ucs):
    return ucs.login_handle.query_dn(distinguished_name)