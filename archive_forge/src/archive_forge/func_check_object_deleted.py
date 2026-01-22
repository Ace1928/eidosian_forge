import configparser
import os
import time
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
def check_object_deleted(self, object_name, object_id, timeout=60):
    """Check that object deleted successfully.

        :param object_name: object name
        :param object_id: uuid4 id of an object
        :param timeout: timeout in seconds
        """
    cmd = self.object_cmd(object_name, 'show')
    try:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if object_id not in self.cinder(cmd, params=object_id):
                break
    except exceptions.CommandFailed:
        pass
    else:
        self.fail('%s %s not deleted after %d seconds.' % (object_name, object_id, timeout))