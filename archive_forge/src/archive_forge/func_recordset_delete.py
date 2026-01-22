import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def recordset_delete(self, zone_id, id, *args, **kwargs):
    cmd = f'recordset delete {zone_id} {id}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)