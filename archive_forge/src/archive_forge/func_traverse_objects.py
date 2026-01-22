from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def traverse_objects(module, ucs, managed_object, mo=''):
    props_match = False
    mo_module = import_module(managed_object['module'])
    mo_class = getattr(mo_module, managed_object['class'])
    if not managed_object['properties'].get('parent_mo_or_dn'):
        managed_object['properties']['parent_mo_or_dn'] = mo
    mo = mo_class(**managed_object['properties'])
    existing_mo = ucs.login_handle.query_dn(mo.dn)
    if module.params['state'] == 'absent':
        if existing_mo:
            if not module.check_mode:
                ucs.login_handle.remove_mo(existing_mo)
            ucs.result['changed'] = True
    else:
        if existing_mo:
            kwargs = dict(managed_object['properties'])
            kwargs.pop('parent_mo_or_dn', None)
            kwargs.pop('pwd', None)
            kwargs.pop('password', None)
            if existing_mo.check_prop_match(**kwargs):
                props_match = True
        if not props_match:
            if not module.check_mode:
                ucs.login_handle.add_mo(mo, modify_present=True)
            ucs.result['changed'] = True
    if managed_object.get('children'):
        for child in managed_object['children']:
            copy_of_child = deepcopy(child)
            traverse_objects(module, ucs, copy_of_child, mo)