import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_award_amount(self, worker_id, assignment_id, amount, reason):
    """
        Update a pairing state to add a task bonus to be paid, appends reason.

        To be used for automatic evaluation bonuses
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        reason = '${} for {}\n'.format(amount, reason)
        cent_amount = int(amount * 100)
        c.execute('UPDATE pairings SET bonus_amount = bonus_amount + ?,\n                        bonus_text = bonus_text || ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (cent_amount, reason, worker_id, assignment_id))
        conn.commit()