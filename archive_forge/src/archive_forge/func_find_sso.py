from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def find_sso(module, name):
    """ Find a SSO using its name """
    path = f'config/sso/idps?name={name}'
    try:
        system = get_system(module)
        sso_result = system.api.get(path=path).get_result()
    except APICommandFailed as err:
        msg = f'Cannot find SSO identity provider {name}: {err}'
        module.fail_json(msg=msg)
    return sso_result