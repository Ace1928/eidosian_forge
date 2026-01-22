from __future__ import absolute_import, division, print_function
import os
import time
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def import_file_to_device(self):
    name = os.path.split(self.want.source)[1]
    self.upload_file_to_device(self.want.source, name)
    time.sleep(2)
    task = self.inline_import()
    self.wait_for_task(task)
    return True