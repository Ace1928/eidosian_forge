import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
def start_syncing(self):
    if self.db:
        self.apply_method(self._really_sync)
    self.sync_timer = threading.Timer(self.sync_period, self.start_syncing)
    self.sync_timer.setDaemon(True)
    self.sync_timer.start()