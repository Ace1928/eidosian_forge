import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def get_assignment_data(self, assignment_id):
    """
        get assignment data for a particular assignment by assignment_id.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM assignments WHERE assignment_id = ?;', (assignment_id,))
        results = c.fetchone()
        return results