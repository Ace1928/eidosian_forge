from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def get_execution_details(self, execid):
    """
        Get the execution details of an API

        Parameters:
            execid (str) - Id for API execution

        Returns:
            response (dict) - Status for API execution
        """
    self.log('Execution Id: {0}'.format(execid), 'DEBUG')
    response = self.dnac._exec(family='task', function='get_business_api_execution_details', params={'execution_id': execid})
    self.log('Response for the current execution: {0}'.format(response))
    return response