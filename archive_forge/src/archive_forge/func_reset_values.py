from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def reset_values(self):
    """Reset all neccessary attributes to default values"""
    self.have.clear()
    self.want.clear()