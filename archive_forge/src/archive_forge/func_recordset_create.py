import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def recordset_create(self, zone_id, name, record=None, type=None, description=None, ttl=None, *args, **kwargs):
    options_str = build_option_string({'--record': record, '--type': type, '--description': description, '--ttl': ttl})
    cmd = f'recordset create {zone_id} {name} {options_str}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)