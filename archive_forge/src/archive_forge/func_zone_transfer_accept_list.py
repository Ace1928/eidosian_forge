import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_transfer_accept_list(self, *args, **kwargs):
    cmd = 'zone transfer accept list'
    return self.parsed_cmd(cmd, ListModel, *args, **kwargs)