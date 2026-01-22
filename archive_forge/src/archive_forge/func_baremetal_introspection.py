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
def baremetal_introspection(self):
    if self._inspector is None:
        self._inspector = self._create_inspector_client()
    return self._inspector