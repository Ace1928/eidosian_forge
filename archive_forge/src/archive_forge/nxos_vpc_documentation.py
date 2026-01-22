from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
peer-keepalive dependency checking.
    1. 'destination' is required with all pkl configs.
    2. If delta has optional pkl keywords present, then all optional pkl
       keywords in existing must be added to delta, otherwise the device cli
       will remove those values when the new config string is issued.
    3. The desired behavior for this set of properties is to merge changes;
       therefore if an optional pkl property exists on the device but not
       in the playbook, then that existing property should be retained.
    Example:
      CLI:       peer-keepalive dest 10.1.1.1 source 10.1.1.2 vrf orange
      Playbook:  {pkl_dest: 10.1.1.1, pkl_vrf: blue}
      Result:    peer-keepalive dest 10.1.1.1 source 10.1.1.2 vrf blue
    