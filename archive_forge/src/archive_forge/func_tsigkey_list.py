import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def tsigkey_list(self, *args, **kwargs):
    return self.parsed_cmd('tsigkey list', ListModel, *args, **kwargs)