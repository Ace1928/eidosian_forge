import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
def process_IN_CREATE(self, raw_event):
    """
        If the event affects a directory and the auto_add flag of the
        targetted watch is set to True, a new watch is added on this
        new directory, with the same attribute values than those of
        this watch.
        """
    if raw_event.mask & IN_ISDIR:
        watch_ = self._watch_manager.get_watch(raw_event.wd)
        created_dir = os.path.join(watch_.path, raw_event.name)
        if watch_.auto_add and (not watch_.exclude_filter(created_dir)):
            addw = self._watch_manager.add_watch
            addw_ret = addw(created_dir, watch_.mask, proc_fun=watch_.proc_fun, rec=False, auto_add=watch_.auto_add, exclude_filter=watch_.exclude_filter)
            created_dir_wd = addw_ret.get(created_dir)
            if created_dir_wd is not None and created_dir_wd > 0 and os.path.isdir(created_dir):
                try:
                    for name in os.listdir(created_dir):
                        inner = os.path.join(created_dir, name)
                        if self._watch_manager.get_wd(inner) is not None:
                            continue
                        if os.path.isfile(inner):
                            flags = IN_CREATE
                        elif os.path.isdir(inner):
                            flags = IN_CREATE | IN_ISDIR
                        else:
                            continue
                        rawevent = _RawEvent(created_dir_wd, flags, 0, name)
                        self._notifier.append_event(rawevent)
                except OSError as err:
                    msg = 'process_IN_CREATE, invalid directory: %s'
                    log.debug(msg % str(err))
    return self.process_default(raw_event)