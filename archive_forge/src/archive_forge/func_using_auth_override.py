import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
@property
def using_auth_override(self):
    return bool(cfg.CONF.identity.override_endpoint)