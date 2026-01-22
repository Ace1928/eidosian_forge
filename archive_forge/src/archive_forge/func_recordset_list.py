import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def recordset_list(self, zone_id, *args, **kwargs):
    cmd = f'recordset list {zone_id}'
    return self.parsed_cmd(cmd, ListModel, *args, **kwargs)