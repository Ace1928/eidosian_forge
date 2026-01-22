import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_worker_note(self, worker_id, assignment_id, note):
    """
        Append a note to the worker notes for a particular worker-assignment pairing.

        Adds newline to the note.
        """
    note += '\n'
    with self.table_access_condition:
        try:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('UPDATE pairings SET notes = notes || ?\n                             WHERE worker_id = ? AND assignment_id = ?;', (note, worker_id, assignment_id))
            conn.commit()
        except Exception as e:
            print(repr(e))