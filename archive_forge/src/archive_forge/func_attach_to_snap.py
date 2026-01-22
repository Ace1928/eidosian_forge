from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def attach_to_snap(self, snapshot, host):
    """ Attach snapshot to a host """
    try:
        if not get_hosts_dict(snapshot):
            snapshot.detach_from(None)
        snapshot.attach_to(host)
        snapshot.update()
    except Exception as e:
        error_msg = 'Failed to attach snapshot [name: %s, id: %s] to host [%s, %s] with error %s' % (snapshot.name, snapshot.id, host.name, host.id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)