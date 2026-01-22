from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def have_required_parameters(self, action):
    """
        Check if all the required parameters in self.params are available or not besides the mandatory parameters
        """
    actions = {'create_aggregate': ['number_of_disks', 'disk_size_size', 'disk_size_unit', 'working_environment_id'], 'update_aggregate': ['number_of_disks', 'disk_size_size', 'disk_size_unit', 'working_environment_id'], 'delete_aggregate': ['working_environment_id']}
    missed_params = [parameter for parameter in actions[action] if parameter not in self.parameters]
    if not missed_params:
        return (True, None)
    else:
        return (False, missed_params)