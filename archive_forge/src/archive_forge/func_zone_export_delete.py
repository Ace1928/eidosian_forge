import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_export_delete(self, zone_export_id, *args, **kwargs):
    cmd = f'zone export delete {zone_export_id}'
    return self.parsed_cmd(cmd, *args, **kwargs)