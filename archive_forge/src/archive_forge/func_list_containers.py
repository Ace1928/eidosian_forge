import configparser as config_parser
import os
from tempest.lib.cli import base
def list_containers(self, params=''):
    return self.zun('container-list', params=params)