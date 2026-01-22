import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_transfer_accept_show(self, id, *args, **kwargs):
    cmd = f'zone transfer accept show {id}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)