import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def share_zone(self, zone_id, target_project_id, *args, **kwargs):
    cmd = f'zone share create {zone_id} {target_project_id}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)