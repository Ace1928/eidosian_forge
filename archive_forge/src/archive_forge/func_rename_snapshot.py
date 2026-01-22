from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from datetime import datetime, timedelta
import time
import copy
def rename_snapshot(self, snapshot_id, new_name):
    """Rename snapshot
            :param snapshot_id: The snapshot id
            :param new_name: The new name of the snapshot
            :return: Boolean indicating if rename operation is successful
        """
    try:
        self.powerflex_conn.volume.rename(snapshot_id, new_name)
        return True
    except Exception as e:
        errormsg = 'Rename snapshot %s operation failed with error %s' % (snapshot_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)