import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_set(self, id, email=None, ttl=None, description=None, type=None, masters=None, *args, **kwargs):
    options_str = build_option_string({'--email': email, '--ttl': ttl, '--description': description, '--masters': masters, '--type': type})
    cmd = f'zone set {id} {options_str}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)