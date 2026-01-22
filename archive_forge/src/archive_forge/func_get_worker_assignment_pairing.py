import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def get_worker_assignment_pairing(self, worker_id, assignment_id):
    """
        get a pairing data structure between a worker and an assignment.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM pairings WHERE worker_id = ?\n                         AND assignment_id = ?;', (worker_id, assignment_id))
        results = c.fetchone()
        return results