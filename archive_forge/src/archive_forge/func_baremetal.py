import logging
import sys
from cliff import app
from cliff import commandmanager
import openstack
from openstack import config as os_config
from osc_lib import utils
import pbr.version
from ironicclient.common import http
from ironicclient.common.i18n import _
from ironicclient import exc
from ironicclient.v1 import client
@property
def baremetal(self):
    if self._ironic is None:
        self._ironic = self._create_ironic_client()
    return self._ironic