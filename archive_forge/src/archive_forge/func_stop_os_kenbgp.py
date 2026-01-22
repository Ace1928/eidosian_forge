import logging
import os
import time
from . import docker_base as base
def stop_os_kenbgp(self, check_running=True, retry=False):
    if check_running:
        if not self.is_running_os_ken():
            return True
    result = False
    if retry:
        try_times = 3
    else:
        try_times = 1
    for _ in range(try_times):
        cmd = '/usr/bin/pkill osken-manager -SIGTERM'
        self.exec_on_ctn(cmd)
        if not self.is_running_os_ken():
            result = True
            break
        time.sleep(1)
    return result