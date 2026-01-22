from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def find_entry(self, entryid):
    results = self.conn.listAllStoragePools()
    if entryid == -1:
        return results
    for entry in results:
        if entry.name() == entryid:
            return entry
    raise EntryNotFound('storage pool %s not found' % entryid)