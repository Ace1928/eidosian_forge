from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def look_up_working_environment_by_name_in_list(self, we_list, name):
    """
        Look up working environment by the name in working environment list
        """
    for we in we_list:
        if we['name'] == name:
            return (we, None)
    return (None, 'look_up_working_environment_by_name_in_list: Working environment not found')