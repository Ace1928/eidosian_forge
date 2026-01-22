import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_new_run(self, target_hits, taskname, task_group_id=None):
    """
        Add a new run to the runs table.
        """
    with self.table_access_condition:
        task_group_id = self._force_task_group_id(task_group_id)
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?);', (task_group_id, 0, target_hits, 0, 0, taskname, time.time()))
        conn.commit()