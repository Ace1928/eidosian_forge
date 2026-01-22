from __future__ import (absolute_import, division, print_function)
import os
import time
import glob
from abc import ABCMeta, abstractmethod
from ansible.module_utils.six import with_metaclass

        method to accept a list of strings as the parameter, find any strings
        in that list that are comma separated, remove them from the list and add
        their comma separated elements to the original list
        