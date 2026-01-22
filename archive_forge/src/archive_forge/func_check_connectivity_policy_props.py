from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def check_connectivity_policy_props(ucs, module, dn):
    props_match = False
    child_dn = dn + '/conn-def'
    mo_1 = ucs.login_handle.query_dn(child_dn)
    if mo_1:
        kwargs = dict(lan_conn_policy_name=module.params['lan_connectivity_policy'])
        kwargs['san_conn_policy_name'] = module.params['san_connectivity_policy']
        if mo_1.check_prop_match(**kwargs):
            props_match = True
    elif not module.params['lan_connectivity_policy'] and (not module.params['san_connectivity_policy']):
        props_match = True
    return props_match