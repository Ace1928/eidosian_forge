import configparser
import os
import time
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
def object_delete(self, object_name, object_id):
    """Delete specified object by ID.

        :param object_name: object name
        :param object_id: uuid4 id of an object
        """
    cmd = self.object_cmd(object_name, 'list')
    cmd_delete = self.object_cmd(object_name, 'delete')
    if object_id in self.cinder(cmd):
        self.cinder(cmd_delete, params=object_id)