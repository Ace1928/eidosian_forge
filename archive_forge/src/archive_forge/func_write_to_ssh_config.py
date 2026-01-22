from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from operator import itemgetter
def write_to_ssh_config(self):
    with open(self.ssh_config_file, 'w+') as f:
        data = self.dump()
        if data:
            f.write(data)
    return self