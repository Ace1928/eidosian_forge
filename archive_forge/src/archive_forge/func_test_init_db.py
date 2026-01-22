import unittest
import os
import time
import uuid
from datetime import datetime
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
from parlai.mturk.core.dev.agents import AssignState
import parlai.mturk.core.dev.mturk_data_handler as DataHandlerFile
def test_init_db(self):
    db_logger = MTurkDataHandler('test1', file_name=self.DB_NAME)
    conn = db_logger._get_connection()
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM runs;')
    self.assertEqual(c.fetchone()[0], 0)
    c.execute('SELECT COUNT(*) FROM hits;')
    self.assertEqual(c.fetchone()[0], 0)
    c.execute('SELECT COUNT(*) FROM assignments;')
    self.assertEqual(c.fetchone()[0], 0)
    c.execute('SELECT COUNT(*) FROM workers;')
    self.assertEqual(c.fetchone()[0], 0)
    c.execute('SELECT COUNT(*) FROM pairings;')
    self.assertEqual(c.fetchone()[0], 0)