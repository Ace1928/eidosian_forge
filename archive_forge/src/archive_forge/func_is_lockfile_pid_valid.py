from __future__ import (absolute_import, division, print_function)
import os
import time
import glob
from abc import ABCMeta, abstractmethod
from ansible.module_utils.six import with_metaclass
@abstractmethod
def is_lockfile_pid_valid(self):
    return