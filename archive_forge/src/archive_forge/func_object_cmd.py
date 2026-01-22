import configparser
import os
import time
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
def object_cmd(self, object_name, cmd):
    return object_name + '-' + cmd if object_name != 'volume' else cmd