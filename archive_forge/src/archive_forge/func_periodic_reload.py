import os
import sys
import time
import threading
import traceback
from paste.util.classinstance import classinstancemethod
def periodic_reload(self):
    while True:
        if not self.check_reload():
            raise SystemRestart()
        time.sleep(self.poll_interval)