import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def get_hits_for_run(self, run_id):
    """
        Get the full list of HITs for the given run_id.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM hits WHERE run_id = ?;', (run_id,))
        results = c.fetchall()
        return results