from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def check_return_status(self):
    """API to check the return status value and exit/fail the module"""
    self.log('status: {0}, msg: {1}'.format(self.status, self.msg), 'DEBUG')
    if 'failed' in self.status:
        self.module.fail_json(msg=self.msg, response=[])
    elif 'exited' in self.status:
        self.module.exit_json(**self.result)
    elif 'invalid' in self.status:
        self.module.fail_json(msg=self.msg, response=[])