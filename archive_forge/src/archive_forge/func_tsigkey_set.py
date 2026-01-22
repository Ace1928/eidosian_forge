import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def tsigkey_set(self, id, name=None, algorithm=None, secret=None, scope=None, *args, **kwargs):
    options_str = build_option_string({'--name': name, '--algorithm': algorithm, '--secret': secret, '--scope': scope})
    cmd = f'tsigkey set {id} {options_str}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)