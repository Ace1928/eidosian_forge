from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from datetime import datetime, timedelta
import time
import copy
def modify_retention(self, snapshot_id, new_retention):
    """Modify snapshot retention
            :param snapshot_id: The snapshot id
            :param new_retention: Desired retention of the snapshot
            :return: Boolean indicating if modifying retention is successful
        """
    try:
        self.powerflex_conn.volume.set_retention_period(snapshot_id, new_retention)
        return True
    except Exception as e:
        errormsg = 'Modify retention of snapshot %s operation failed with error %s' % (snapshot_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)