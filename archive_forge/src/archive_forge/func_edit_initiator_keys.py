from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def edit_initiator_keys(host_initiators, include_key_list):
    """
    For each host initiator, remove keys not in the include_key_list.
    For FCs, add a long address. This is the address with colons inserted.
    Return the edited host initiators list.
    """
    trimmed_initiators = []
    for init in host_initiators:
        if init['type'] == 'FC' and 'address' in init.keys():
            address_str = str(init['address'])
            address_iter = iter(address_str)
            long_address = ':'.join((a + b for a, b in zip(address_iter, address_iter)))
            init['address_long'] = long_address
        trimmed_item = {key: val for key, val in init.items() if key in include_key_list}
        trimmed_initiators.append(trimmed_item)
    return trimmed_initiators