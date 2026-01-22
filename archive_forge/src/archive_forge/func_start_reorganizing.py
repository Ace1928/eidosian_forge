import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
def start_reorganizing(self):
    if not self.max_age:
        return
    self.log.debug('reorganizing the database')
    breakpoint = datetime.datetime.now() - datetime.timedelta(seconds=self.max_age)
    db = self._get_new_connection()
    c = db.cursor()
    try:
        c.execute('DELETE FROM %s WHERE r_updated<%%s' % self.table_name, (breakpoint,))
    except (MySQLdb.Error, AttributeError) as e:
        self.log.warn('Unable to reorganise: %s', e)
    finally:
        c.close()
        db.close()
    self.reorganize_timer = threading.Timer(self.reorganize_period, self.start_reorganizing)
    self.reorganize_timer.setDaemon(True)
    self.reorganize_timer.start()