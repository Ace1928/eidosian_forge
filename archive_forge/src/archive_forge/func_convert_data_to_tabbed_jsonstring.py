from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
@staticmethod
def convert_data_to_tabbed_jsonstring(data):
    """
        Convert a dictionary data to json format string
        """
    dump = json.dumps(data, indent=2, separators=(',', ': '))
    return re.sub('\n +', lambda match: '\n' + '\t' * int(len(match.group().strip('\n')) / 2), dump)