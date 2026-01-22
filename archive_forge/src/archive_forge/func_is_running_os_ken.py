import logging
import os
import time
from . import docker_base as base
def is_running_os_ken(self):
    results = self.exec_on_ctn('ps ax')
    running = False
    for line in results.split('\n')[1:]:
        if 'osken-manager' in line:
            running = True
    return running