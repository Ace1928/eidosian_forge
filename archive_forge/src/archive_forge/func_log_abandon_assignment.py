import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_abandon_assignment(self, worker_id, assignment_id):
    """
        To be called whenever a worker returns a hit.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE assignments SET status = ?\n                         WHERE assignment_id = ?;', ('Abandoned', assignment_id))
        c.execute('UPDATE workers SET disconnected = disconnected + 1\n                         WHERE worker_id = ?;', (worker_id,))
        conn.commit()