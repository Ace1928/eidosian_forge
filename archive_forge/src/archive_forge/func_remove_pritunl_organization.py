from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.net_tools.pritunl.api import (
def remove_pritunl_organization(module):
    result = {}
    org_name = module.params.get('name')
    force = module.params.get('force')
    org_obj_list = []
    org_obj_list = list_pritunl_organizations(**dict_merge(get_pritunl_settings(module), {'filters': {'name': org_name}}))
    if len(org_obj_list) == 0:
        result['changed'] = False
        result['response'] = {}
    else:
        org = org_obj_list[0]
        if force or org['user_count'] == 0:
            response = delete_pritunl_organization(**dict_merge(get_pritunl_settings(module), {'organization_id': org['id']}))
            result['changed'] = True
            result['response'] = response
        else:
            module.fail_json(msg="Can not remove organization '%s' with %d attached users. Either set 'force' option to true or remove active users from the organization" % (org_name, org['user_count']))
    module.exit_json(**result)