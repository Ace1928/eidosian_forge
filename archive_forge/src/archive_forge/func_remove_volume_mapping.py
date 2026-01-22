from __future__ import (absolute_import, division, print_function)
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleError
def remove_volume_mapping(self, name, host):
    """remove volume mapping to record table (luns_by_target)."""
    for host_group in self.array_facts['netapp_host_groups']:
        if host == host_group['name']:
            for entry in self.luns_by_target[host_group['name']]:
                if entry[0] == name:
                    del entry
            for hostgroup_host in host_group['hosts']:
                for entry in self.luns_by_target[hostgroup_host]:
                    if entry[0] == name:
                        del entry
            break
    else:
        for index, entry in enumerate(self.luns_by_target[host]):
            if entry[0] == name:
                self.luns_by_target[host].pop(index)