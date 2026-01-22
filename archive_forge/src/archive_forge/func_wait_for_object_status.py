import configparser
import os
import time
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
def wait_for_object_status(self, object_name, object_id, status, timeout=120, interval=3):
    """Wait until object reaches given status.

        :param object_name: object name
        :param object_id: uuid4 id of an object
        :param status: expected status of an object
        :param timeout: timeout in seconds
        """
    cmd = self.object_cmd(object_name, 'show')
    start_time = time.time()
    while time.time() - start_time < timeout:
        if status in self.cinder(cmd, params=object_id):
            break
        time.sleep(interval)
    else:
        self.fail('%s %s did not reach status %s after %d seconds.' % (object_name, object_id, status, timeout))