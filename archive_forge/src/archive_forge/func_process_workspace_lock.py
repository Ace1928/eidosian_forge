from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def process_workspace_lock(self):
    self.conn.process_workspace_locking(self.module.params)