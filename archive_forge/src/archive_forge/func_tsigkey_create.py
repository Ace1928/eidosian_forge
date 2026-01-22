import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def tsigkey_create(self, name, algorithm, secret, scope, resource_id, *args, **kwargs):
    options_str = build_option_string({'--name': name, '--algorithm': algorithm, '--secret': secret, '--scope': scope, '--resource-id': resource_id})
    cmd = f'tsigkey create {options_str}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)