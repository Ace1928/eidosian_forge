import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_transfer_accept_request(self, id, key, *args, **kwargs):
    options_str = build_option_string({'--transfer-id': id, '--key': key})
    cmd = f'zone transfer accept request {options_str}'
    return self.parsed_cmd(cmd, *args, **kwargs)