from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def endpoint_list_spec():
    return dict(provider=dict(type='dict', options=endpoint_argument_spec()), metrics=dict(type='dict', options=endpoint_argument_spec()), alerts=dict(type='dict', options=endpoint_argument_spec()), ssh_keypair=dict(type='dict', options=endpoint_argument_spec(), no_log=False))