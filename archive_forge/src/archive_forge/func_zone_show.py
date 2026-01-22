import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_show(self, id, *args, **kwargs):
    return self.parsed_cmd(f'zone show {id}', FieldValueModel, *args, **kwargs)