from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def make_mo_dict(ucs_mo):
    obj_dict = {}
    for mo_property in ucs_mo.prop_map.values():
        obj_dict[mo_property] = getattr(ucs_mo, mo_property)
    return obj_dict