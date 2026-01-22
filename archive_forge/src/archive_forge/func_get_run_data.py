import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def get_run_data(self, task_group_id):
    """
        get the run data for the given task_group_id, return None if not found.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM runs WHERE run_id = ?;', (task_group_id,))
        results = c.fetchone()
        return results